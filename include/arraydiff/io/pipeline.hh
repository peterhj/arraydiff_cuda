#ifndef ARRAYDIFF_IO_PIPELINE_HH
#define ARRAYDIFF_IO_PIPELINE_HH

#include <boost/lockfree/spsc_queue.hpp>

#include <atomic>
#include <cassert>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <experimental/optional>
#include <functional>
#include <iostream>
#include <memory>
#include <random>
#include <thread>
#include <utility>

namespace arraydiff {
namespace io {

using std::atomic;
using std::experimental::optional;
using std::function;
using std::make_shared;
using std::shared_ptr;
using std::vector;

class Datum {
public:
  Datum(const uint8_t* ptr, size_t sz)
    : buffer(ptr), size(sz) {}

  const uint8_t* buffer;
  size_t size;
};

template <typename T = uint8_t>
class MutDatum {
public:
  static MutDatum Alloc(size_t sz) {
    T* buf = (T*)std::malloc(sizeof(T) * sz);
    return MutDatum(buf, sz);
  }

  MutDatum()
    : buffer(NULL), size(0) {}
  MutDatum(T* ptr, size_t sz)
    : buffer(ptr), size(sz) {}

  void dealloc() {
    std::free(this->buffer);
    this->buffer = NULL;
    this->size = 0;
  }

  T* buffer;
  size_t size;
};

template <typename Item>
class RandomAccessData {
public:
  virtual ~RandomAccessData() {}

  virtual size_t size() const = 0;
  virtual Item at(size_t index) const = 0;
};

template <typename Item>
class DataSource {
public:
  virtual ~DataSource() {}

  virtual Item next() = 0;
};

template <typename Item>
class CyclicDataSource : public virtual DataSource<Item> {
public:
  explicit CyclicDataSource(shared_ptr<RandomAccessData<Item>> data)
    : data_(data), counter_(0UL) {}
  virtual ~CyclicDataSource() {}

  virtual Item next() {
    size_t index = counter_;
    assert(index < data_->size());
    ++counter_;
    if (counter_ >= data_->size()) {
      counter_ = 0UL;
    }
    return data_->at(index);
  }

private:
  shared_ptr<RandomAccessData<Item>> data_;
  size_t counter_;
};

template <typename Item>
class PermuteDataSource : public virtual DataSource<Item> {
public:
  explicit PermuteDataSource(shared_ptr<RandomAccessData<Item>> data)
    : data_(data), permuted_(), counter_(0UL)
  {
    for (size_t i = 0; i < data_->size(); ++i) {
      permuted_.push_back(i);
    }
    assert(permuted_.size() == data_->size());
    if (data_->size() > 1) {
      // Use Fisher-Yates shuffle to generate a permutation.
      std::random_device rd;
      std::mt19937_64 rng(rd());
      for (size_t i = 0; i < data_->size() - 1; ++i) {
        std::uniform_int_distribution<size_t> dist_j(i, data_->size() - 1);
        size_t j = dist_j(rng);
        if (i != j) {
          auto tmp_i = permuted_.at(i);
          permuted_.at(i) = permuted_.at(j);
          permuted_.at(j) = tmp_i;
        }
      }
    }
  }

  virtual ~PermuteDataSource() {}

  virtual Item next() {
    size_t index = permuted_.at(counter_);
    assert(index < data_->size());
    ++counter_;
    if (counter_ >= data_->size()) {
      counter_ = 0UL;
    }
    return data_->at(index);
  }

private:
  shared_ptr<RandomAccessData<Item>> data_;
  vector<size_t> permuted_;
  size_t counter_;
};

template <typename Item>
class RandomSamplingDataSource : public virtual DataSource<Item> {
public:
  explicit RandomSamplingDataSource(shared_ptr<RandomAccessData<Item>> data)
    : data_(data), dist_(0UL, data->size() - 1UL)
  {
    std::random_device rd;
    rng_ = std::mt19937_64(rd());
  }
  virtual ~RandomSamplingDataSource() {}

  virtual Item next() {
    size_t index = dist_(rng_);
    assert(index < data_->size());
    return data_->at(index);
  }

private:
  shared_ptr<RandomAccessData<Item>> data_;
  std::uniform_int_distribution<size_t> dist_;
  std::mt19937_64 rng_;
};

template <typename Item>
class RandomPermuteDataSource : public virtual DataSource<Item> {
public:
  explicit RandomPermuteDataSource(shared_ptr<RandomAccessData<Item>> data)
    : data_(data), permuted_(), counter_(-1)
  {
    std::random_device rd;
    rng_ = std::mt19937_64(rd());
    for (size_t i = 0; i < data_->size(); ++i) {
      permuted_.push_back(i);
    }
    assert(permuted_.size() == data_->size());
  }

  virtual ~RandomPermuteDataSource() {}

  virtual Item next() {
    if (counter_ >= data_->size()) {
      std::clog << "DEBUG: RandomPermuteDataSource:"
          << " counter: " << counter_
          << " resetting..." << std::endl;
      // Use Fisher-Yates shuffle to generate a permutation.
      assert(permuted_.size() == data_->size());
      for (size_t i = 0; i < data_->size() - 1; ++i) {
        std::uniform_int_distribution<size_t> dist_j(i, data_->size() - 1);
        size_t j = dist_j(rng_);
        if (i != j) {
          auto tmp_i = permuted_.at(i);
          permuted_.at(i) = permuted_.at(j);
          permuted_.at(j) = tmp_i;
        }
      }
      counter_ = 0;
    }
    size_t index = permuted_.at(counter_);
    assert(index < data_->size());
    if (counter_ < 1000UL) {
      std::clog << "DEBUG: counter: " << counter_ << " index: " << index << std::endl;
    }
    ++counter_;
    return data_->at(index);
  }

private:
  shared_ptr<RandomAccessData<Item>> data_;
  std::mt19937_64 rng_;
  vector<size_t> permuted_;
  size_t counter_;
};

template <typename Item>
class RandomJoinDataSource /*: public virtual DataSource<Item>*/ {
public:
  explicit RandomJoinDataSource(vector<shared_ptr<DataSource<Item>>> srcs);
  //virtual ~RandomJoinDataSource() {}
};

template <typename Item>
using DataPipelineQueue = boost::lockfree::spsc_queue<Item, boost::lockfree::capacity<1024>>;

template <typename SrcItem, typename Item>
class DataPipelineWorker {
public:
  DataPipelineWorker(
      size_t worker_rank,
      size_t num_workers,
      shared_ptr<DataSource<SrcItem>> source,
      function<Item(size_t, size_t, SrcItem, std::mt19937_64*)> transform,
      shared_ptr<DataPipelineQueue<Item>> queue,
      shared_ptr<atomic<bool>> stop)
    : worker_rank_(worker_rank),
      num_workers_(num_workers),
      source_(source),
      transform_(transform),
      queue_(queue),
      stop_(stop) {}

  void run_loop() {
    std::random_device rd;
    std::mt19937_64 rng(rd());
    volatile bool stop_signal = false;
    for (; !stop_signal; stop_signal = stop_->load()) {
      SrcItem src_item = source_->next();
      Item item = (transform_)(worker_rank_, num_workers_, src_item, &rng);
      bool pushed = false;
      if (!stop_signal && queue_->push(item)) {
        pushed = true;
      }
      if (!pushed) {
        for (size_t t1 = 0; t1 < 10UL && !stop_signal; ++t1) {
          for (size_t t2 = 0; t2 < 100UL && !stop_signal; ++t2) {
            stop_signal = stop_->load();
          }
          if (!stop_signal && queue_->push(item)) {
            pushed = true;
            break;
          }
        }
      }
      if (!pushed) {
        while (!stop_signal) {
          std::this_thread::sleep_for(std::chrono::milliseconds(1));
          stop_signal = stop_->load();
          if (!stop_signal && queue_->push(item)) {
            pushed = true;
            break;
          }
        }
      }
    }
  }

private:
  size_t worker_rank_;
  size_t num_workers_;
  shared_ptr<DataSource<SrcItem>> source_;
  function<Item(size_t, size_t, SrcItem, std::mt19937_64*)> transform_;
  shared_ptr<DataPipelineQueue<Item>> queue_;
  shared_ptr<atomic<bool>> stop_;
};

template <typename Item>
class DataPipelineDataSource : public virtual DataSource<optional<Item>> {
public:
  template <typename SrcItem>
  DataPipelineDataSource(
      size_t num_workers,
      function<shared_ptr<DataSource<SrcItem>>(void)> source_gen,
      function<Item(size_t, size_t, SrcItem, std::mt19937_64*)> transform)
    : num_workers_(num_workers), counter_(0)
  {
    assert(num_workers_ >= 1);
    stop_ = make_shared<atomic<bool>>(false);
    for (size_t p = 0; p < num_workers_; ++p) {
      auto source = source_gen();
      auto queue = make_shared<DataPipelineQueue<Item>>();
      auto thread = make_shared<std::thread>([=]() {
        DataPipelineWorker<SrcItem, Item> worker(p, num_workers_, source, transform, queue, stop_);
        worker.run_loop();
      });
      queue_.push_back(queue);
      thread_.push_back(thread);
    }
  }

  virtual ~DataPipelineDataSource() {
    stop_->store(true);
    for (size_t p = 0; p < num_workers_; ++p) {
      thread_[p]->join();
    }
  }

  virtual optional<Item> next() {
    size_t queue_id = counter_ % num_workers_;
    Item item;
    while (!queue_[queue_id]->pop(item));
    ++counter_;
    return item;
  }

private:
  size_t num_workers_;
  //function<DataSource<SrcItem>(void)> source_gen_;
  size_t counter_;
  vector<shared_ptr<std::thread>> thread_;
  vector<shared_ptr<DataPipelineQueue<Item>>> queue_;
  shared_ptr<atomic<bool>> stop_;
};

} // namespace io
} // namespace arraydiff

#endif
