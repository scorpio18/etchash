// Copyright 2017 The go-ethereum Authors
// This file is part of the go-ethereum library.
//
// The go-ethereum library is free software: you can redistribute it and/or modify
// it under the terms of the GNU Lesser General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// The go-ethereum library is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with the go-ethereum library. If not, see <http://www.gnu.org/licenses/>.

// Package etchash implements the etchash proof-of-work consensus engine.
package etchash

import (
	"errors"
	"fmt"
	"math"
	"math/big"
	"math/rand"
	"os"
	"path/filepath"
	"reflect"
	"runtime"
	"strconv"
	"sync"
	"sync/atomic"
	"time"
	"unsafe"

	"github.com/edsrzf/mmap-go"
	lrupkg "github.com/ethereum/go-ethereum/common/lru"
	"github.com/ethereum/go-ethereum/consensus"
	"github.com/ethereum/go-ethereum/log"
	"github.com/ethereum/go-ethereum/metrics"
	"github.com/ethereum/go-ethereum/rpc"
)

var ErrInvalidDumpMagic = errors.New("invalid dump magic")

var (
	// two256 is a big integer representing 2^256
	two256 = new(big.Int).Exp(big.NewInt(2), big.NewInt(256), big.NewInt(0))

	// sharedEthash is a full instance that can be shared between multiple users.
	sharedEthash *Etchash

	// algorithmRevision is the data structure version used for file naming.
	algorithmRevision = 23

	// dumpMagic is a dataset dump header to sanity check a data dump.
	dumpMagic = []uint32{0xbaddcafe, 0xfee1dead}
)

func init() {
	sharedConfig := Config{
		PowMode:       ModeNormal,
		CachesInMem:   3,
		DatasetsInMem: 1,
	}
	sharedEthash = New(sharedConfig, nil, false)
}

// isLittleEndian returns whether the local system is running in little or big
// endian byte order.
func isLittleEndian() bool {
	n := uint32(0x01020304)
	return *(*byte)(unsafe.Pointer(&n)) == 0x04
}

// memoryMap tries to memory map a file of uint32s for read only access.
func memoryMap(path string, lock bool) (*os.File, mmap.MMap, []uint32, error) {
	file, err := os.OpenFile(path, os.O_RDONLY, 0644)
	if err != nil {
		return nil, nil, nil, err
	}
	mem, buffer, err := memoryMapFile(file, false)
	if err != nil {
		file.Close()
		return nil, nil, nil, err
	}
	for i, magic := range dumpMagic {
		if buffer[i] != magic {
			mem.Unmap()
			file.Close()
			return nil, nil, nil, ErrInvalidDumpMagic
		}
	}
	if lock {
		if err := mem.Lock(); err != nil {
			mem.Unmap()
			file.Close()
			return nil, nil, nil, err
		}
	}
	return file, mem, buffer[len(dumpMagic):], err
}

// memoryMapFile tries to memory map an already opened file descriptor.
func memoryMapFile(file *os.File, write bool) (mmap.MMap, []uint32, error) {
	// Try to memory map the file
	flag := mmap.RDONLY
	if write {
		flag = mmap.RDWR
	}
	mem, err := mmap.Map(file, flag, 0)
	if err != nil {
		return nil, nil, err
	}
	// The file is now memory-mapped. Create a []uint32 view of the file.
	var view []uint32
	header := (*reflect.SliceHeader)(unsafe.Pointer(&view))
	header.Data = (*reflect.SliceHeader)(unsafe.Pointer(&mem)).Data
	header.Cap = len(mem) / 4
	header.Len = header.Cap
	return mem, view, nil
}

// memoryMapAndGenerate tries to memory map a temporary file of uint32s for write
// access, fill it with the data from a generator and then move it into the final
// path requested.
func memoryMapAndGenerate(path string, size uint64, lock bool, generator func(buffer []uint32)) (*os.File, mmap.MMap, []uint32, error) {
	// Ensure the data folder exists
	if err := os.MkdirAll(filepath.Dir(path), 0755); err != nil {
		return nil, nil, nil, err
	}
	// Create a huge temporary empty file to fill with data
	temp := path + "." + strconv.Itoa(rand.Int())

	dump, err := os.Create(temp)
	if err != nil {
		return nil, nil, nil, err
	}
	if err = ensureSize(dump, int64(len(dumpMagic))*4+int64(size)); err != nil {
		dump.Close()
		os.Remove(temp)
		return nil, nil, nil, err
	}
	// Memory map the file for writing and fill it with the generator
	mem, buffer, err := memoryMapFile(dump, true)
	if err != nil {
		dump.Close()
		os.Remove(temp)
		return nil, nil, nil, err
	}
	copy(buffer, dumpMagic)

	data := buffer[len(dumpMagic):]
	generator(data)

	if err := mem.Unmap(); err != nil {
		return nil, nil, nil, err
	}
	if err := dump.Close(); err != nil {
		return nil, nil, nil, err
	}
	if err := os.Rename(temp, path); err != nil {
		return nil, nil, nil, err
	}
	return memoryMap(path, lock)
}

type cacheOrDataset interface {
	*cache | *dataset
}

// lru tracks caches or datasets by their last use time, keeping at most N of them.
type lru[T cacheOrDataset] struct {
	what string
	new  func(epoch uint64) T
	mu   sync.Mutex
	// Items are kept in a LRU cache, but there is a special case:
	// We always keep an item for (highest seen epoch) + 1 as the 'future item'.
	cache      lrupkg.BasicLRU[uint64, T]
	future     uint64
	futureItem T
}

// newlru create a new least-recently-used cache for either the verification caches
// or the mining datasets.
func newlru[T cacheOrDataset](maxItems int, new func(epoch uint64) T) *lru[T] {
	var what string
	switch any(T(nil)).(type) {
	case *cache:
		what = "cache"
	case *dataset:
		what = "dataset"
	default:
		panic("unknown type")
	}
	return &lru[T]{
		what:  what,
		new:   new,
		cache: lrupkg.NewBasicLRU[uint64, T](maxItems),
	}
}

// get retrieves or creates an item for the given epoch. The first return value is always
// non-nil. The second return value is non-nil if lru thinks that an item will be useful in
// the near future.
func (lru *lru[T]) get(epoch uint64) (item, future T) {
	lru.mu.Lock()
	defer lru.mu.Unlock()

	// Get or create the item for the requested epoch.
	item, ok := lru.cache.Get(epoch)
	if !ok {
		if lru.future > 0 && lru.future == epoch {
			item = lru.futureItem
		} else {
			log.Trace("Requiring new etchash "+lru.what, "epoch", epoch)
			item = lru.new(epoch)
		}
		lru.cache.Add(epoch, item)
	}
	// Update the 'future item' if epoch is larger than previously seen.
	if epoch < maxEpoch-1 && lru.future < epoch+1 {
		log.Trace("Requiring new future etchash "+lru.what, "epoch", epoch+1)
		future = lru.new(epoch + 1)
		lru.future = epoch + 1
		lru.futureItem = future
	}
	return item, future
}

// cache wraps an etchash cache with some metadata to allow easier concurrent use.
type cache struct {
	epoch uint64    // Epoch for which this cache is relevant
	dump  *os.File  // File descriptor of the memory mapped cache
	mmap  mmap.MMap // Memory map itself to unmap before releasing
	cache []uint32  // The actual cache data content (may be memory mapped)
	once  sync.Once // Ensures the cache is generated only once
}

// newCache creates a new etchash verification cache.
func newCache(epoch uint64) *cache {
	return &cache{epoch: epoch}
}

// generate ensures that the cache content is generated before use.
func (c *cache) generate(dir string, limit int, lock bool, test bool) {
	c.once.Do(func() {
		size := cacheSize(c.epoch*epochLength + 1)
		seed := seedHash(c.epoch*epochLength + 1)
		if test {
			size = 1024
		}
		// If we don't store anything on disk, generate and return.
		if dir == "" {
			c.cache = make([]uint32, size/4)
			generateCache(c.cache, c.epoch, seed)
			return
		}
		// Disk storage is needed, this will get fancy
		var endian string
		if !isLittleEndian() {
			endian = ".be"
		}
		path := filepath.Join(dir, fmt.Sprintf("cache-R%d-%x%s", algorithmRevision, seed[:8], endian))
		logger := log.New("epoch", c.epoch)

		// We're about to mmap the file, ensure that the mapping is cleaned up when the
		// cache becomes unused.
		runtime.SetFinalizer(c, (*cache).finalizer)

		// Try to load the file from disk and memory map it
		var err error
		c.dump, c.mmap, c.cache, err = memoryMap(path, lock)
		if err == nil {
			logger.Debug("Loaded old etchash cache from disk")
			return
		}
		logger.Debug("Failed to load old etchash cache", "err", err)

		// No previous cache available, create a new cache file to fill
		c.dump, c.mmap, c.cache, err = memoryMapAndGenerate(path, size, lock, func(buffer []uint32) { generateCache(buffer, c.epoch, seed) })
		if err != nil {
			logger.Error("Failed to generate mapped etchash cache", "err", err)

			c.cache = make([]uint32, size/4)
			generateCache(c.cache, c.epoch, seed)
		}
		// Iterate over all previous instances and delete old ones
		for ep := int(c.epoch) - limit; ep >= 0; ep-- {
			seed := seedHash(uint64(ep)*epochLength + 1)
			path := filepath.Join(dir, fmt.Sprintf("cache-R%d-%x%s*", algorithmRevision, seed[:8], endian))
			files, _ := filepath.Glob(path) // find also the temp files that are generated.
			for _, file := range files {
				os.Remove(file)
			}
		}
	})
}

// finalizer unmaps the memory and closes the file.
func (c *cache) finalizer() {
	if c.mmap != nil {
		c.mmap.Unmap()
		c.dump.Close()
		c.mmap, c.dump = nil, nil
	}
}

// dataset wraps an etchash dataset with some metadata to allow easier concurrent use.
type dataset struct {
	epoch   uint64      // Epoch for which this cache is relevant
	dump    *os.File    // File descriptor of the memory mapped cache
	mmap    mmap.MMap   // Memory map itself to unmap before releasing
	dataset []uint32    // The actual cache data content
	once    sync.Once   // Ensures the cache is generated only once
	done    atomic.Bool // Atomic flag to determine generation status
}

// newDataset creates a new etchash mining dataset and returns it as a plain Go
// interface to be usable in an LRU cache.
func newDataset(epoch uint64) *dataset {
	return &dataset{epoch: epoch}
}

// generate ensures that the dataset content is generated before use.
func (d *dataset) generate(dir string, limit int, lock bool, test bool) {
	d.once.Do(func() {
		// Mark the dataset generated after we're done. This is needed for remote
		defer d.done.Store(true)

		csize := cacheSize(d.epoch*epochLength + 1)
		dsize := datasetSize(d.epoch*epochLength + 1)
		seed := seedHash(d.epoch*epochLength + 1)
		if test {
			csize = 1024
			dsize = 32 * 1024
		}
		// If we don't store anything on disk, generate and return
		if dir == "" {
			cache := make([]uint32, csize/4)
			generateCache(cache, d.epoch, seed)

			d.dataset = make([]uint32, dsize/4)
			generateDataset(d.dataset, d.epoch, cache)

			return
		}
		// Disk storage is needed, this will get fancy
		var endian string
		if !isLittleEndian() {
			endian = ".be"
		}
		path := filepath.Join(dir, fmt.Sprintf("full-R%d-%x%s", algorithmRevision, seed[:8], endian))
		logger := log.New("epoch", d.epoch)

		// We're about to mmap the file, ensure that the mapping is cleaned up when the
		// cache becomes unused.
		runtime.SetFinalizer(d, (*dataset).finalizer)

		// Try to load the file from disk and memory map it
		var err error
		d.dump, d.mmap, d.dataset, err = memoryMap(path, lock)
		if err == nil {
			logger.Debug("Loaded old etchash dataset from disk")
			return
		}
		logger.Debug("Failed to load old etchash dataset", "err", err)

		// No previous dataset available, create a new dataset file to fill
		cache := make([]uint32, csize/4)
		generateCache(cache, d.epoch, seed)

		d.dump, d.mmap, d.dataset, err = memoryMapAndGenerate(path, dsize, lock, func(buffer []uint32) { generateDataset(buffer, d.epoch, cache) })
		if err != nil {
			logger.Error("Failed to generate mapped etchash dataset", "err", err)

			d.dataset = make([]uint32, dsize/4)
			generateDataset(d.dataset, d.epoch, cache)
		}
		// Iterate over all previous instances and delete old ones
		for ep := int(d.epoch) - limit; ep >= 0; ep-- {
			seed := seedHash(uint64(ep)*epochLength + 1)
			path := filepath.Join(dir, fmt.Sprintf("full-R%d-%x%s", algorithmRevision, seed[:8], endian))
			os.Remove(path)
		}
	})
}

// generated returns whether this particular dataset finished generating already
// or not (it may not have been started at all). This is useful for remote miners
// to default to verification caches instead of blocking on DAG generations.
func (d *dataset) generated() bool {
	return d.done.Load()
}

// finalizer closes any file handlers and memory maps open.
func (d *dataset) finalizer() {
	if d.mmap != nil {
		d.mmap.Unmap()
		d.dump.Close()
		d.mmap, d.dump = nil, nil
	}
}

// MakeCache generates a new etchash cache and optionally stores it to disk.
func MakeCache(block uint64, dir string) {
	c := cache{epoch: block / epochLength}
	c.generate(dir, math.MaxInt32, false, false)
}

// MakeDataset generates a new etchash dataset and optionally stores it to disk.
func MakeDataset(block uint64, dir string) {
	d := dataset{epoch: block / epochLength}
	d.generate(dir, math.MaxInt32, false, false)
}

// Mode defines the type and amount of PoW verification an etchash engine makes.
type Mode uint

const (
	ModeNormal Mode = iota
	ModeShared
	ModeTest
	ModeFake
	ModeFullFake
)

// Config are the configuration parameters of the etchash.
type Config struct {
	CacheDir         string
	CachesInMem      int
	CachesOnDisk     int
	CachesLockMmap   bool
	DatasetDir       string
	DatasetsInMem    int
	DatasetsOnDisk   int
	DatasetsLockMmap bool
	PowMode          Mode

	// When set, notifications sent by the remote sealer will
	// be block header JSON objects instead of work package arrays.
	NotifyFull bool

	Log log.Logger `toml:"-"`
}

// Etchash is a consensus engine based on proof-of-work implementing the etchash
// algorithm.
type Etchash struct {
	config Config

	caches   *lru[*cache]   // In memory caches to avoid regenerating too often
	datasets *lru[*dataset] // In memory datasets to avoid regenerating too often

	// Mining related fields
	rand     *rand.Rand    // Properly seeded random source for nonces
	threads  int           // Number of threads to mine on if mining
	update   chan struct{} // Notification channel to update mining parameters
	hashrate metrics.Meter // Meter tracking the average hashrate
	remote   *remoteSealer

	// The fields below are hooks for testing
	shared    *Etchash       // Shared PoW verifier to avoid cache regeneration
	fakeFail  uint64        // Block number which fails PoW check even in fake mode
	fakeDelay time.Duration // Time delay to sleep for before returning from verify

	lock      sync.Mutex // Ensures thread safety for the in-memory caches and mining fields
	closeOnce sync.Once  // Ensures exit channel will not be closed twice.
}

// New creates a full sized etchash PoW scheme and starts a background thread for
// remote mining, also optionally notifying a batch of remote services of new work
// packages.
func New(config Config, notify []string, noverify bool) *Etchash {
	if config.Log == nil {
		config.Log = log.Root()
	}
	if config.CachesInMem <= 0 {
		config.Log.Warn("One etchash cache must always be in memory", "requested", config.CachesInMem)
		config.CachesInMem = 1
	}
	if config.CacheDir != "" && config.CachesOnDisk > 0 {
		config.Log.Info("Disk storage enabled for etchash caches", "dir", config.CacheDir, "count", config.CachesOnDisk)
	}
	if config.DatasetDir != "" && config.DatasetsOnDisk > 0 {
		config.Log.Info("Disk storage enabled for etchash DAGs", "dir", config.DatasetDir, "count", config.DatasetsOnDisk)
	}
	etchash := &Etchash{
		config:   config,
		caches:   newlru(config.CachesInMem, newCache),
		datasets: newlru(config.DatasetsInMem, newDataset),
		update:   make(chan struct{}),
		hashrate: metrics.NewMeterForced(),
	}
	if config.PowMode == ModeShared {
		etchash.shared = sharedEthash
	}
	etchash.remote = startRemoteSealer(etchash, notify, noverify)
	return etchash
}

// NewTester creates a small sized etchash PoW scheme useful only for testing
// purposes.
func NewTester(notify []string, noverify bool) *Etchash {
	return New(Config{PowMode: ModeTest}, notify, noverify)
}

// NewFaker creates a etchash consensus engine with a fake PoW scheme that accepts
// all blocks' seal as valid, though they still have to conform to the Ethereum
// consensus rules.
func NewFaker() *Etchash {
	return &Etchash{
		config: Config{
			PowMode: ModeFake,
			Log:     log.Root(),
		},
	}
}

// NewFakeFailer creates a etchash consensus engine with a fake PoW scheme that
// accepts all blocks as valid apart from the single one specified, though they
// still have to conform to the Ethereum consensus rules.
func NewFakeFailer(fail uint64) *Etchash {
	return &Etchash{
		config: Config{
			PowMode: ModeFake,
			Log:     log.Root(),
		},
		fakeFail: fail,
	}
}

// NewFakeDelayer creates a etchash consensus engine with a fake PoW scheme that
// accepts all blocks as valid, but delays verifications by some time, though
// they still have to conform to the Ethereum consensus rules.
func NewFakeDelayer(delay time.Duration) *Etchash {
	return &Etchash{
		config: Config{
			PowMode: ModeFake,
			Log:     log.Root(),
		},
		fakeDelay: delay,
	}
}

// NewFullFaker creates an etchash consensus engine with a full fake scheme that
// accepts all blocks as valid, without checking any consensus rules whatsoever.
func NewFullFaker() *Etchash {
	return &Etchash{
		config: Config{
			PowMode: ModeFullFake,
			Log:     log.Root(),
		},
	}
}

// NewShared creates a full sized etchash PoW shared between all requesters running
// in the same process.
func NewShared() *Etchash {
	return &Etchash{shared: sharedEthash}
}

// Close closes the exit channel to notify all backend threads exiting.
func (etchash *Etchash) Close() error {
	return etchash.StopRemoteSealer()
}

// StopRemoteSealer stops the remote sealer
func (etchash *Etchash) StopRemoteSealer() error {
	etchash.closeOnce.Do(func() {
		// Short circuit if the exit channel is not allocated.
		if etchash.remote == nil {
			return
		}
		close(etchash.remote.requestExit)
		<-etchash.remote.exitCh
	})
	return nil
}

// cache tries to retrieve a verification cache for the specified block number
// by first checking against a list of in-memory caches, then against caches
// stored on disk, and finally generating one if none can be found.
func (etchash *Etchash) cache(block uint64) *cache {
	epoch := block / epochLength
	current, future := etchash.caches.get(epoch)

	// Wait for generation finish.
	current.generate(etchash.config.CacheDir, etchash.config.CachesOnDisk, etchash.config.CachesLockMmap, etchash.config.PowMode == ModeTest)

	// If we need a new future cache, now's a good time to regenerate it.
	if future != nil {
		go future.generate(etchash.config.CacheDir, etchash.config.CachesOnDisk, etchash.config.CachesLockMmap, etchash.config.PowMode == ModeTest)
	}
	return current
}

// dataset tries to retrieve a mining dataset for the specified block number
// by first checking against a list of in-memory datasets, then against DAGs
// stored on disk, and finally generating one if none can be found.
//
// If async is specified, not only the future but the current DAG is also
// generates on a background thread.
func (etchash *Etchash) dataset(block uint64, async bool) *dataset {
	// Retrieve the requested etchash dataset
	epoch := block / epochLength
	current, future := etchash.datasets.get(epoch)

	// If async is specified, generate everything in a background thread
	if async && !current.generated() {
		go func() {
			current.generate(etchash.config.DatasetDir, etchash.config.DatasetsOnDisk, etchash.config.DatasetsLockMmap, etchash.config.PowMode == ModeTest)
			if future != nil {
				future.generate(etchash.config.DatasetDir, etchash.config.DatasetsOnDisk, etchash.config.DatasetsLockMmap, etchash.config.PowMode == ModeTest)
			}
		}()
	} else {
		// Either blocking generation was requested, or already done
		current.generate(etchash.config.DatasetDir, etchash.config.DatasetsOnDisk, etchash.config.DatasetsLockMmap, etchash.config.PowMode == ModeTest)
		if future != nil {
			go future.generate(etchash.config.DatasetDir, etchash.config.DatasetsOnDisk, etchash.config.DatasetsLockMmap, etchash.config.PowMode == ModeTest)
		}
	}
	return current
}

// Threads returns the number of mining threads currently enabled. This doesn't
// necessarily mean that mining is running!
func (etchash *Etchash) Threads() int {
	etchash.lock.Lock()
	defer etchash.lock.Unlock()

	return etchash.threads
}

// SetThreads updates the number of mining threads currently enabled. Calling
// this method does not start mining, only sets the thread count. If zero is
// specified, the miner will use all cores of the machine. Setting a thread
// count below zero is allowed and will cause the miner to idle, without any
// work being done.
func (etchash *Etchash) SetThreads(threads int) {
	etchash.lock.Lock()
	defer etchash.lock.Unlock()

	// If we're running a shared PoW, set the thread count on that instead
	if etchash.shared != nil {
		etchash.shared.SetThreads(threads)
		return
	}
	// Update the threads and ping any running seal to pull in any changes
	etchash.threads = threads
	select {
	case etchash.update <- struct{}{}:
	default:
	}
}

// Hashrate implements PoW, returning the measured rate of the search invocations
// per second over the last minute.
// Note the returned hashrate includes local hashrate, but also includes the total
// hashrate of all remote miner.
func (etchash *Etchash) Hashrate() float64 {
	// Short circuit if we are run the etchash in normal/test mode.
	if etchash.config.PowMode != ModeNormal && etchash.config.PowMode != ModeTest {
		return etchash.hashrate.Rate1()
	}
	var res = make(chan uint64, 1)

	select {
	case etchash.remote.fetchRateCh <- res:
	case <-etchash.remote.exitCh:
		// Return local hashrate only if etchash is stopped.
		return etchash.hashrate.Rate1()
	}

	// Gather total submitted hash rate of remote sealers.
	return etchash.hashrate.Rate1() + float64(<-res)
}

// APIs implements consensus.Engine, returning the user facing RPC APIs.
func (etchash *Etchash) APIs(chain consensus.ChainHeaderReader) []rpc.API {
	// In order to ensure backward compatibility, we exposes etchash RPC APIs
	// to both eth and etchash namespaces.
	return []rpc.API{
		{
			Namespace: "eth",
			Service:   &API{etchash},
		},
		{
			Namespace: "etchash",
			Service:   &API{etchash},
		},
	}
}

// SeedHash is the seed to use for generating a verification cache and the mining
// dataset.
func SeedHash(block uint64) []byte {
	return seedHash(block)
}
