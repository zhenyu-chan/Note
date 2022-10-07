# cyber

## base

### HashMap

数组+链表

template <typename K, typename V, std::size_t TableSize = 128,
          typename std::enable_if<std::is_integral<K>::value &&
                                      (TableSize & (TableSize - 1)) == 0,
                                  int>::type = 0>
class AtomicHashMap

K必须为 int 类型且size必须为2的幂

    Bucket table_[TableSize];
    uint64_t capacity_;
    uint64_t mode_num_;

```
template <typename K, typename V, std::size_t TableSize = 128,
          typename std::enable_if<std::is_integral<K>::value &&
                                      (TableSize & (TableSize - 1)) == 0,
                                  int>::type = 0>
class AtomicHashMap


  AtomicHashMap(const AtomicHashMap &other) = delete;
  AtomicHashMap &operator=(const AtomicHashMap &other) = delete;

  struct Entry {

    K key = 0;
    std::atomic<V *> value_ptr = {nullptr};
    std::atomic<Entry *> next = {nullptr};
  };

  class Bucket {

    void Insert(K key, const V &value) {
      Entry *prev = nullptr;
      Entry *target = nullptr;
      Entry *new_entry = nullptr;
      V *new_value = nullptr;
      while (true) {
        if (Find(key, &prev, &target)) {
          // key exists, update value
          if (!new_value) {
            new_value = new V(value);
          }
          auto old_val_ptr = target->value_ptr.load(std::memory_order_acquire);
          if (target->value_ptr.compare_exchange_strong(
                  old_val_ptr, new_value, std::memory_order_acq_rel,
                  std::memory_order_relaxed)) {
            delete old_val_ptr;
            if (new_entry) {
              delete new_entry;
              new_entry = nullptr;
            }
            return;
          }
          continue;
        } else {
          if (!new_entry) {
            new_entry = new Entry(key, value);
          }
          new_entry->next.store(target, std::memory_order_release);
          if (prev->next.compare_exchange_strong(target, new_entry,
                                                 std::memory_order_acq_rel,
                                                 std::memory_order_relaxed)) {
            // Insert success
            if (new_value) {
              delete new_value;
              new_value = nullptr;
            }
            return;
          }
          // another entry has been inserted, retry
        }
      }
    }

    Entry *head_;
  };

  Bucket table_[TableSize];
  uint64_t capacity_;
  uint64_t mode_num_;
```

#### Entry

节点实现封装，值和链表链接原子操作atomic
K——键       K key
V——值       std::atomic<V *>  value_ptr
链表链接    std::atomic<Entry *> next

添加节点memory_order_release store，释放节点memory_order_acquire load

#### Bucket

实现链表并按Key升序排列

  Entry *head_;


### rw_lock_guard

对锁进行封装实现锁的自动加锁和解锁（RAII）
```
template <typename RWLock>
class ReadLockGuard {
 public:
  explicit ReadLockGuard(RWLock& lock) : rw_lock_(lock) { rw_lock_.ReadLock(); }

  ~ReadLockGuard() { rw_lock_.ReadUnlock(); }

 private:
  ReadLockGuard(const ReadLockGuard& other) = delete;
  ReadLockGuard& operator=(const ReadLockGuard& other) = delete;
  RWLock& rw_lock_;
};

template <typename RWLock>
class WriteLockGuard {
 public:
  explicit WriteLockGuard(RWLock& lock) : rw_lock_(lock) {
    rw_lock_.WriteLock();
  }

  ~WriteLockGuard() { rw_lock_.WriteUnlock(); }

 private:
  WriteLockGuard(const WriteLockGuard& other) = delete;
  WriteLockGuard& operator=(const WriteLockGuard& other) = delete;
  RWLock& rw_lock_;
};
```

### AtomicRWLock 

lock_num_对锁引用计数，读锁+1，写锁-1；当lock_num_小于0说明有写锁，write_lock_wait_num_等待写锁的数量
```
class AtomicRWLock {
  friend class ReadLockGuard<AtomicRWLock>;
  friend class WriteLockGuard<AtomicRWLock>;

 public:
  static const int32_t RW_LOCK_FREE = 0;          //读写分界点
  static const int32_t WRITE_EXCLUSIVE = -1;      //写标志点
  static const uint32_t MAX_RETRY_TIMES = 5;      //最大重试次数，若还未获得锁则yeild
  AtomicRWLock() {}
  explicit AtomicRWLock(bool write_first) : write_first_(write_first) {}

 private:
  // all these function only can used by ReadLockGuard/WriteLockGuard;
  void ReadLock();
  void WriteLock();

  void ReadUnlock();
  void WriteUnlock();

  AtomicRWLock(const AtomicRWLock&) = delete;
  AtomicRWLock& operator=(const AtomicRWLock&) = delete;
  std::atomic<uint32_t> write_lock_wait_num_ = {0};
  std::atomic<int32_t> lock_num_ = {0};
  bool write_first_ = true;
};
```

### reentrant_rw_lock

可重入的读写锁
记录写线程的pid，加/解读写锁加一层判断是否当前线程为写线程
若当前线程为写线程则读锁失败，加写锁直接-1（即写线程可加多重写锁）/解读锁若lock_num_为-1则写线程为默认初始值（小于-1则直接返回说明锁依旧持有在写线程）

```
static const std::thread::id NULL_THREAD_ID = std::thread::id();
class ReentrantRWLock {
  friend class ReadLockGuard<ReentrantRWLock>;
  friend class WriteLockGuard<ReentrantRWLock>;

 public:
  static const int32_t RW_LOCK_FREE = 0;
  static const int32_t WRITE_EXCLUSIVE = -1;
  static const uint32_t MAX_RETRY_TIMES = 5;
  static const std::thread::id null_thread;
  ReentrantRWLock() {}
  explicit ReentrantRWLock(bool write_first) : write_first_(write_first) {}

 private:
  // all these function only can used by ReadLockGuard/WriteLockGuard;
  void ReadLock();
  void WriteLock();

  void ReadUnlock();
  void WriteUnlock();

  ReentrantRWLock(const ReentrantRWLock&) = delete;
  ReentrantRWLock& operator=(const ReentrantRWLock&) = delete;
  std::thread::id write_thread_id_ = {NULL_THREAD_ID};
  std::atomic<uint32_t> write_lock_wait_num_ = {0};
  std::atomic<int32_t> lock_num_ = {0};
  bool write_first_ = true;
};
```

### macro

//该宏用于判断name模板类中T是否有func这一成员函数

```
#define CACHELINE_SIZE 64
#define DEFINE_TYPE_TRAIT(name, func)                     \
  template <typename T>                                   \
  struct name {                                           \
    template <typename Class>                             \
    static constexpr bool Test(decltype(&Class::func)*) { \
      return true;                                        \
    }                                                     \
    template <typename>                                   \
    static constexpr bool Test(...) {                     \
      return false;                                       \
    }                                                     \
                                                          \
    static constexpr bool value = Test<T>(nullptr);       \
  };                                                      \
                                                          \
  template <typename T>                                   \
  constexpr bool name<T>::value;

inline void cpu_relax() {
#if defined(__aarch64__)
  asm volatile("yield" ::: "memory");
#else
  asm volatile("rep; nop" ::: "memory");
#endif

void* CheckedMalloc(size_t size)
void* CheckedCalloc(size_t num, size_t size) 

```


### wait_strategy

阻塞策略，默认条件变量通知，还有sleep阻塞等待、yeild非阻塞等待、忙等待直接返回、带超时的条件变量
```
class WaitStrategy {
 public:
  virtual void NotifyOne() {}
  virtual void BreakAllWait() {}
  virtual bool EmptyWait() = 0;
  virtual ~WaitStrategy() {}
};

class BlockWaitStrategy : public WaitStrategy {
 public:
  BlockWaitStrategy() {}
  void NotifyOne() override { cv_.notify_one(); }

  bool EmptyWait() override {
    std::unique_lock<std::mutex> lock(mutex_);
    cv_.wait(lock);
    return true;
  }

  void BreakAllWait() override { cv_.notify_all(); }

 private:
  std::mutex mutex_;
  std::condition_variable cv_;
};

class SleepWaitStrategy : public WaitStrategy {
 public:
  SleepWaitStrategy() {}
  explicit SleepWaitStrategy(uint64_t sleep_time_us)
      : sleep_time_us_(sleep_time_us) {}

  bool EmptyWait() override {
    std::this_thread::sleep_for(std::chrono::microseconds(sleep_time_us_));
    return true;
  }

  void SetSleepTimeMicroSeconds(uint64_t sleep_time_us) {
    sleep_time_us_ = sleep_time_us;
  }

 private:
  uint64_t sleep_time_us_ = 10000;
};

class YieldWaitStrategy : public WaitStrategy {
 public:
  YieldWaitStrategy() {}
  bool EmptyWait() override {
    std::this_thread::yield();
    return true;
  }
};

class BusySpinWaitStrategy : public WaitStrategy {
 public:
  BusySpinWaitStrategy() {}
  bool EmptyWait() override { return true; }
};

class TimeoutBlockWaitStrategy : public WaitStrategy {
 public:
  TimeoutBlockWaitStrategy() {}
  explicit TimeoutBlockWaitStrategy(uint64_t timeout)
      : time_out_(std::chrono::milliseconds(timeout)) {}

  void NotifyOne() override { cv_.notify_one(); }

  bool EmptyWait() override {
    std::unique_lock<std::mutex> lock(mutex_);
    if (cv_.wait_for(lock, time_out_) == std::cv_status::timeout) {
      return false;
    }
    return true;
  }

  void BreakAllWait() override { cv_.notify_all(); }

  void SetTimeout(uint64_t timeout) {
    time_out_ = std::chrono::milliseconds(timeout);
  }

 private:
  std::mutex mutex_;
  std::condition_variable cv_;
  std::chrono::milliseconds time_out_;
};
```

### bounded_queue

头尾节点各占一个空间（pool_size = size + 2），分配连续的内存以数组管理
头尾节点不重合，commit为头节点之后一个位置，可与尾节点重合
enqueue在原尾节点构造新元素，commit后移成功后notifyone
dequeue在原头节点后一个位置取元素，头节点与commit不能相同
wait系列即调用EmptyWait接口，等资源释放即notifyone
```
template <typename T>
class BoundedQueue {
 public:
  using value_type = T;
  using size_type = uint64_t;

 public:
  BoundedQueue() {}
  BoundedQueue& operator=(const BoundedQueue& other) = delete;
  BoundedQueue(const BoundedQueue& other) = delete;
  ~BoundedQueue();
  bool Init(uint64_t size);
  bool Init(uint64_t size, WaitStrategy* strategy);
  bool Enqueue(const T& element);
  bool Enqueue(T&& element);
  bool WaitEnqueue(const T& element);
  bool WaitEnqueue(T&& element);
  bool Dequeue(T* element);
  bool WaitDequeue(T* element);
  uint64_t Size();
  bool Empty();
  void SetWaitStrategy(WaitStrategy* WaitStrategy);
  void BreakAllWait();
  uint64_t Head() { return head_.load(); }
  uint64_t Tail() { return tail_.load(); }
  uint64_t Commit() { return commit_.load(); }

 private:
  uint64_t GetIndex(uint64_t num);
  // return num - (num / pool_size_) * pool_size_;  // faster than %

  alignas(CACHELINE_SIZE) std::atomic<uint64_t> head_ = {0};
  alignas(CACHELINE_SIZE) std::atomic<uint64_t> tail_ = {1};
  alignas(CACHELINE_SIZE) std::atomic<uint64_t> commit_ = {1};
  // alignas(CACHELINE_SIZE) std::atomic<uint64_t> size_ = {0};
  uint64_t pool_size_ = 0;
  T* pool_ = nullptr;
  std::unique_ptr<WaitStrategy> wait_strategy_ = nullptr;
  volatile bool break_all_wait_ = false;
};
```

### unbounded_queue

链表实现，空间不连续
Node链表节点，数据+引用计数+next，引用初始值为2（head和tail或者pre->next和tail）
head不存数据，tail存数据，每次dequeue时候释放head并移动head到下一节点
```
template <typename T>
class UnboundedQueue {
 public:
  UnboundedQueue() { Reset(); }
  UnboundedQueue& operator=(const UnboundedQueue& other) = delete;
  UnboundedQueue(const UnboundedQueue& other) = delete;

  ~UnboundedQueue() { Destroy(); }

  void Clear() {
    Destroy();
    Reset();
  }

  void Enqueue(const T& element) {
    auto node = new Node();
    node->data = element;
    Node* old_tail = tail_.load();

    while (true) {
      if (tail_.compare_exchange_strong(old_tail, node)) {
        old_tail->next = node;
        old_tail->release();
        size_.fetch_add(1);
        break;
      }
    }
  }

  bool Dequeue(T* element) {
    Node* old_head = head_.load();
    Node* head_next = nullptr;
    do {
      head_next = old_head->next;

      if (head_next == nullptr) {
        return false;
      }
    } while (!head_.compare_exchange_strong(old_head, head_next));
    *element = head_next->data;
    size_.fetch_sub(1);
    old_head->release();
    return true;
  }

  size_t Size() { return size_.load(); }

  bool Empty() { return size_.load() == 0; }

 private:
  struct Node {
    T data;
    std::atomic<uint32_t> ref_count;
    Node* next = nullptr;
    Node() { ref_count.store(2); }
    void release() {
      ref_count.fetch_sub(1);
      if (ref_count.load() == 0) {
        delete this;
      }
    }
  };

  void Reset() {
    auto node = new Node();
    head_.store(node);
    tail_.store(node);
    size_.store(0);
  }

  void Destroy() {
    auto ite = head_.load();
    Node* tmp = nullptr;
    while (ite != nullptr) {
      tmp = ite->next;
      delete ite;
      ite = tmp;
    }
  }

  std::atomic<Node*> head_;
  std::atomic<Node*> tail_;
  std::atomic<size_t> size_;
};
```


### thread_safe_queue

std::queue 线程不安全，需要加锁
```
template <typename T>
class ThreadSafeQueue {
 public:
  ThreadSafeQueue() {}
  ThreadSafeQueue& operator=(const ThreadSafeQueue& other) = delete;
  ThreadSafeQueue(const ThreadSafeQueue& other) = delete;

  ~ThreadSafeQueue() { BreakAllWait(); }

  void Enqueue(const T& element) {
    std::lock_guard<std::mutex> lock(mutex_);
    queue_.emplace(element);
    cv_.notify_one();
  }

  bool Dequeue(T* element) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (queue_.empty()) {
      return false;
    }
    *element = std::move(queue_.front());
    queue_.pop();
    return true;
  }

  bool WaitDequeue(T* element) {
    std::unique_lock<std::mutex> lock(mutex_);
    cv_.wait(lock, [this]() { return break_all_wait_ || !queue_.empty(); });
    if (break_all_wait_) {
      return false;
    }
    *element = std::move(queue_.front());
    queue_.pop();
    return true;
  }

  typename std::queue<T>::size_type Size() {
    std::lock_guard<std::mutex> lock(mutex_);
    return queue_.size();
  }

  bool Empty() {
    std::lock_guard<std::mutex> lock(mutex_);
    return queue_.empty();
  }

  void BreakAllWait() {
    break_all_wait_ = true;
    cv_.notify_all();
  }

 private:
  volatile bool break_all_wait_ = false;
  std::mutex mutex_;
  std::queue<T> queue_;
  std::condition_variable cv_;
};
```

### for_each

LessThan即有小于运算直接比较否则使用不等于计算
```
DEFINE_TYPE_TRAIT(HasLess, operator<)  // NOLINT

template <class Value, class End>
typename std::enable_if<HasLess<Value>::value && HasLess<End>::value,
                        bool>::type
LessThan(const Value& val, const End& end) {
  return val < end;
}

template <class Value, class End>
typename std::enable_if<!HasLess<Value>::value || !HasLess<End>::value,
                        bool>::type
LessThan(const Value& val, const End& end) {
  return val != end;
}

#define FOR_EACH(i, begin, end)           \
  for (auto i = (true ? (begin) : (end)); \
       apollo::cyber::base::LessThan(i, (end)); ++i)
```


### concurrent_object

Node即objet和next指针，Head即count和Node指针
node_arena_内存池指针（Node），连续内存，但使用过程中由于内存的释放导致内存使用并不能按照数组使用
类似boost的object_pool
实际过程中count似乎并无用处
```
template <typename T>
class CCObjectPool : public std::enable_shared_from_this<CCObjectPool<T>> {
 public:
  explicit CCObjectPool(uint32_t size);
  virtual ~CCObjectPool();

  template <typename... Args>
  void ConstructAll(Args &&... args);

  template <typename... Args>
  std::shared_ptr<T> ConstructObject(Args &&... args);

  std::shared_ptr<T> GetObject();
  void ReleaseObject(T *);
  uint32_t size() const;

 private:
  struct Node {
    T object;
    Node *next;
  };

  struct alignas(2 * sizeof(Node *)) Head {
    uintptr_t count;
    Node *node;
  };

 private:
  CCObjectPool(CCObjectPool &) = delete;
  CCObjectPool &operator=(CCObjectPool &) = delete;
  bool FindFreeHead(Head *head);

  std::atomic<Head> free_head_;
  Node *node_arena_ = nullptr;
  uint32_t capacity_ = 0;
};
```

### object_pool

free_head指向链表头（链表与数组逆序  <-）,其他与concurrent_object_pool类似
```
template <typename T>
class ObjectPool : public std::enable_shared_from_this<ObjectPool<T>> {
 public:
  using InitFunc = std::function<void(T *)>;
  using ObjectPoolPtr = std::shared_ptr<ObjectPool<T>>;

  template <typename... Args>
  explicit ObjectPool(uint32_t num_objects, Args &&... args);

  template <typename... Args>
  ObjectPool(uint32_t num_objects, InitFunc f, Args &&... args);

  virtual ~ObjectPool();

  std::shared_ptr<T> GetObject();

 private:
  struct Node {
    T object;
    Node *next;
  };

  ObjectPool(ObjectPool &) = delete;
  ObjectPool &operator=(ObjectPool &) = delete;
  void ReleaseObject(T *);

  uint32_t num_objects_ = 0;
  char *object_arena_ = nullptr;
  Node *free_head_ = nullptr;
};
```

### thread_pool

利用future、return_type、packaged_task实现线程池，线程池是vector、任务池是BoundedQueue
```
class ThreadPool {
 public:
  explicit ThreadPool(std::size_t thread_num, std::size_t max_task_num = 1000);

  template <typename F, typename... Args>
  auto Enqueue(F&& f, Args&&... args)
      -> std::future<typename std::result_of<F(Args...)>::type>;

  ~ThreadPool();

 private:
  std::vector<std::thread> workers_;
  BoundedQueue<std::function<void()>> task_queue_;
  std::atomic_bool stop_;
};

inline ThreadPool::ThreadPool(std::size_t threads, std::size_t max_task_num)
    : stop_(false) {
  if (!task_queue_.Init(max_task_num, new BlockWaitStrategy())) {
    throw std::runtime_error("Task queue init failed.");
  }
  workers_.reserve(threads);
  for (size_t i = 0; i < threads; ++i) {
    workers_.emplace_back([this] {
      while (!stop_) {
        std::function<void()> task;
        if (task_queue_.WaitDequeue(&task)) {
          task();
        }
      }
    });
  }
}

// before using the return value, you should check value.valid()
template <typename F, typename... Args>
auto ThreadPool::Enqueue(F&& f, Args&&... args)
    -> std::future<typename std::result_of<F(Args...)>::type> {
  using return_type = typename std::result_of<F(Args...)>::type;

  auto task = std::make_shared<std::packaged_task<return_type()>>(
      std::bind(std::forward<F>(f), std::forward<Args>(args)...));

  std::future<return_type> res = task->get_future();

  // don't allow enqueueing after stopping the pool
  if (stop_) {
    return std::future<return_type>();
  }
  task_queue_.Enqueue([task]() { (*task)(); });
  return res;
};

// the destructor joins all threads
inline ThreadPool::~ThreadPool() {
  if (stop_.exchange(true)) {
    return;
  }
  task_queue_.BreakAllWait();
  for (std::thread& worker : workers_) {
    worker.join();
  }
}
```

### signal

Signal、Connection、Slot
Slot即函数指针Callback和状态标志connected_
Connection即SlotPtr（Slot智能指针）和SignalPtr（Signal对象）
```
```

## blocker

### blocker

publish存储消息，并将消息通过published_callbacks_调用回调函数处理消息
observe即Observe()函数直接取publish队列所有消息
subscribe即添加回调函数
published_callbacks_为call_id和回调函数指针的unordered_map
```
class BlockerBase {
 public:
  virtual ~BlockerBase() = default;

  virtual void Reset() = 0;
  virtual void ClearObserved() = 0;
  virtual void ClearPublished() = 0;
  virtual void Observe() = 0;
  virtual bool IsObservedEmpty() const = 0;
  virtual bool IsPublishedEmpty() const = 0;
  virtual bool Unsubscribe(const std::string& callback_id) = 0;

  virtual size_t capacity() const = 0;
  virtual void set_capacity(size_t capacity) = 0;
  virtual const std::string& channel_name() const = 0;
};

struct BlockerAttr {
  BlockerAttr() : capacity(10), channel_name("") {}
  explicit BlockerAttr(const std::string& channel)
      : capacity(10), channel_name(channel) {}
  BlockerAttr(size_t cap, const std::string& channel)
      : capacity(cap), channel_name(channel) {}
  BlockerAttr(const BlockerAttr& attr)
      : capacity(attr.capacity), channel_name(attr.channel_name) {}

  size_t capacity;
  std::string channel_name;
};

template <typename T>
class Blocker : public BlockerBase {
  friend class BlockerManager;

 public:
  using MessageType = T;
  using MessagePtr = std::shared_ptr<T>;
  using MessageQueue = std::list<MessagePtr>;
  using Callback = std::function<void(const MessagePtr&)>;
  using CallbackMap = std::unordered_map<std::string, Callback>;
  using Iterator = typename std::list<std::shared_ptr<T>>::const_iterator;

  explicit Blocker(const BlockerAttr& attr);
  virtual ~Blocker();

  void Publish(const MessageType& msg);
  void Publish(const MessagePtr& msg);

  void ClearObserved() override;
  void ClearPublished() override;
  void Observe() override;
  bool IsObservedEmpty() const override;
  bool IsPublishedEmpty() const override;

  bool Subscribe(const std::string& callback_id, const Callback& callback);
  bool Unsubscribe(const std::string& callback_id) override;

  const MessageType& GetLatestObserved() const;
  const MessagePtr GetLatestObservedPtr() const;
  const MessagePtr GetOldestObservedPtr() const;
  const MessagePtr GetLatestPublishedPtr() const;

  Iterator ObservedBegin() const;
  Iterator ObservedEnd() const;

  size_t capacity() const override;
  void set_capacity(size_t capacity) override;
  const std::string& channel_name() const override;

 private:
  void Reset() override;
  void Enqueue(const MessagePtr& msg);
  void Notify(const MessagePtr& msg);

  BlockerAttr attr_;
  MessageQueue observed_msg_queue_;
  MessageQueue published_msg_queue_;
  mutable std::mutex msg_mutex_;

  CallbackMap published_callbacks_;
  mutable std::mutex cb_mutex_;

  MessageType dummy_msg_;
};
```

### blocker_manager

blockers_为channel_name和blocker的unordered_map映射
```
class BlockerManager {
 public:
  using BlockerMap =
      std::unordered_map<std::string, std::shared_ptr<BlockerBase>>;

  virtual ~BlockerManager();

  static const std::shared_ptr<BlockerManager>& Instance() {
    static auto instance =
        std::shared_ptr<BlockerManager>(new BlockerManager());
    return instance;
  }

  template <typename T>
  bool Publish(const std::string& channel_name,
               const typename Blocker<T>::MessagePtr& msg);

  template <typename T>
  bool Publish(const std::string& channel_name,
               const typename Blocker<T>::MessageType& msg);

  template <typename T>
  bool Subscribe(const std::string& channel_name, size_t capacity,
                 const std::string& callback_id,
                 const typename Blocker<T>::Callback& callback);

  template <typename T>
  bool Unsubscribe(const std::string& channel_name,
                   const std::string& callback_id);

  template <typename T>
  std::shared_ptr<Blocker<T>> GetBlocker(const std::string& channel_name);

  template <typename T>
  std::shared_ptr<Blocker<T>> GetOrCreateBlocker(const BlockerAttr& attr);

  void Observe();
  void Reset();

 private:
  BlockerManager();
  BlockerManager(const BlockerManager&) = delete;
  BlockerManager& operator=(const BlockerManager&) = delete;

  BlockerMap blockers_;
  std::mutex blocker_mutex_;
};
```

### intra_writer


### intra_reader


## class_loader


## common

### environment

```
std::string GetEnv(const std::string& var_name,const std::string& default_value = "")
//获取var_name环境变量，若无则返回default_value

inline const std::string WorkRoot() {
  std::string work_root = GetEnv("CYBER_PATH");
  if (work_root.empty()) {
    work_root = "/apollo/cyber";
  }
  return work_root;
}
//获取工作目录

```

### file

二进制、ASCII、protobuf之间转换以及目录、文件等相关判断

### global_data

单例，提供node、channel、service哈希表以及其他参数

```
  // global config
  CyberConfig config_;
  // host info
  std::string host_ip_;
  std::string host_name_;
  // process info
  int process_id_;
  std::string process_group_;
  int component_nums_ = 0;
  // sched policy info
  std::string sched_name_ = "CYBER_DEFAULT";
  // run mode
  RunMode run_mode_;
  ClockMode clock_mode_;
  static AtomicHashMap<uint64_t, std::string, 512> node_id_map_;
  static AtomicHashMap<uint64_t, std::string, 256> channel_id_map_;
  static AtomicHashMap<uint64_t, std::string, 256> service_id_map_;
  static AtomicHashMap<uint64_t, std::string, 256> task_id_map_;
```

```
namespace {
const std::string& kEmptyString = "";
std::string program_path() {
  char path[PATH_MAX];
  auto len = readlink("/proc/self/exe", path, sizeof(path) - 1);
  if (len == -1) {
    return kEmptyString;
  }
  path[len] = '\0';
  return std::string(path);
}
}  // namespace
```

### log

日志记录宏，基于glog

### macro

T类型有Shutdown成员函数则HasShutdown<T>::value为true否则为false
```
DEFINE_TYPE_TRAIT(HasShutdown, Shutdown)

template <typename T>
typename std::enable_if<HasShutdown<T>::value>::type CallShutdown(T *instance) {
  instance->Shutdown();
}

template <typename T>
typename std::enable_if<!HasShutdown<T>::value>::type CallShutdown(
    T *instance) {
  (void)instance;
}
```

禁止拷贝和赋值运算

```
#define DISALLOW_COPY_AND_ASSIGN(classname) \
  classname(const classname &) = delete;    \
  classname &operator=(const classname &) = delete;
```

单例，std::once_flag和call_once实现,主动销毁CleanUp() 调用类CallShutdown

```
#define DECLARE_SINGLETON(classname)                                      \
 public:                                                                  \
  static classname *Instance(bool create_if_needed = true) {              \
    static classname *instance = nullptr;                                 \
    if (!instance && create_if_needed) {                                  \
      static std::once_flag flag;                                         \
      std::call_once(flag,                                                \
                     [&] { instance = new (std::nothrow) classname(); }); \
    }                                                                     \
    return instance;                                                      \
  }                                                                       \
                                                                          \
  static void CleanUp() {                                                 \
    auto instance = Instance(false);                                      \
    if (instance != nullptr) {                                            \
      CallShutdown(instance);                                             \
    }                                                                     \
  }                                                                       \
                                                                          \
 private:                                                                 \
  classname();                                                            \
  DISALLOW_COPY_AND_ASSIGN(classname)
```

### time_conversion

解决**闰秒**问题,实现unix时间和GPS时间相互转换
```
// UNIX time counts seconds since 1970-1-1, without leap seconds.
// GPS time counts seconds since 1980-1-6, with leap seconds.
// When a leap second is inserted, UNIX time is ambiguous, as shown below.
//    UNIX date and time      UNIX epoch     GPS epoch
//    2008-12-31 23:59:59.0   1230767999.0   914803213.0
//    2008-12-31 23:59:59.5   1230767999.5   914803213.5
//    2008-12-31 23:59:60.0   1230768000.0   914803214.0
//    2008-12-31 23:59:60.5   1230768000.5   914803214.5
//    2009-01-01 00:00:00.0   1230768000.0   914803215.0
//    2009-01-01 00:00:00.5   1230768000.5   914803215.5

// A table of when a leap second is inserted and cumulative leap seconds.
static const std::vector<std::pair<int32_t, int32_t>> LEAP_SECONDS = {
    // UNIX time, leap seconds
    // Add future leap seconds here.
    {1483228800, 18},  // 2017-01-01
    {1435708800, 17},  // 2015-07-01
    {1341100800, 16},  // 2012-07-01
    {1230768000, 15},  // 2009-01-01
    {1136073600, 14},  // 2006-01-01
};

constexpr int32_t UNIX_GPS_DIFF = 315964800;
```

时间戳与string互转
```
inline uint64_t StringToUnixSeconds(
    const std::string& time_str,
    const std::string& format_str = "%Y-%m-%d %H:%M:%S") {
  tm tmp_time;
  strptime(time_str.c_str(), format_str.c_str(), &tmp_time);
  tmp_time.tm_isdst = -1;
  time_t time = mktime(&tmp_time);
  return static_cast<uint64_t>(time);
}

inline std::string UnixSecondsToString(
    uint64_t unix_seconds,
    const std::string& format_str = "%Y-%m-%d-%H:%M:%S") {
  std::time_t t = unix_seconds;
  struct tm ptm;
  struct tm* ret = localtime_r(&t, &ptm);
  if (ret == nullptr) {
    return std::string("");
  }
  uint32_t length = 64;
  std::vector<char> buff(length, '\0');
  strftime(buff.data(), length, format_str.c_str(), ret);
  return std::string(buff.data());
}
```

### types

```
class NullType {};

// Return code definition for cyber internal function return.
enum ReturnCode {
  SUCC = 0,
  FAIL = 1,
};

/**
 * @brief Describe relation between nodes, writers/readers...
 */
enum Relation : std::uint8_t {
  NO_RELATION = 0,
  DIFF_HOST,  // different host
  DIFF_PROC,  // same host, but different process
  SAME_PROC,  // same process
};

static const char SRV_CHANNEL_REQ_SUFFIX[] = "__SRV__REQUEST";
static const char SRV_CHANNEL_RES_SUFFIX[] = "__SRV__RESPONSE";
```

### util

hash生成key

```
inline std::size_t Hash(const std::string& key) {
  return std::hash<std::string>{}(key);
}
```

强类型[enum](https://blog.csdn.net/u010487568/article/details/53643876)
转化为其对应的底层类型
```
template <typename Enum>
auto ToInt(Enum const value) -> typename std::underlying_type<Enum>::type {
  return static_cast<typename std::underlying_type<Enum>::type>(value);
}
```


## component


## conf


## croutine

//https://blog.csdn.net/jinzhuojun/article/details/86760743

swap_x86_x64.S,保存协程栈并切换
x86_64寄存器中传参少于7个时，依次放入rdi、rsi、rcx、rdx、r8、r9
若大于7个，前6个不变,以8个为例则8(%esp)、(%esp)即存在栈上后续从栈上取出
rax第一个返回寄存器

//  The stack layout looks as follows:
//
//              +------------------+
//              |      Reserved    |
//              +------------------+
//              |  Return Address  |   f1
//              +------------------+
//              |        RDI       |   arg
//              +------------------+
//              |        R12       |
//              +------------------+
//              |        R13       |
//              +------------------+
//              |        ...       |
//              +------------------+
// ctx->sp  =>  |        RBP       |
//              +------------------+

```
constexpr size_t STACK_SIZE = 2 * 1024 * 1024;
#if defined __aarch64__
constexpr size_t REGISTERS_SIZE = 160;
#else
constexpr size_t REGISTERS_SIZE = 56;
#endif

typedef void (*func)(void*);
struct RoutineContext {
  char stack[STACK_SIZE];
  char* sp = nullptr;
#if defined __aarch64__
} __attribute__((aligned(16)));
#else
};
#endif

void MakeContext(const func &f1, const void *arg, RoutineContext *ctx) {
  ctx->sp = ctx->stack + STACK_SIZE - 2 * sizeof(void *) - REGISTERS_SIZE;
  std::memset(ctx->sp, 0, REGISTERS_SIZE);
#ifdef __aarch64__
  char *sp = ctx->stack + STACK_SIZE - sizeof(void *);
#else
  char *sp = ctx->stack + STACK_SIZE - 2 * sizeof(void *);
#endif
  *reinterpret_cast<void **>(sp) = reinterpret_cast<void *>(f1);
  sp -= sizeof(void *);
  *reinterpret_cast<void **>(sp) = const_cast<void *>(arg);
}

inline void SwapContext(char** src_sp, char** dest_sp) {
  ctx_swap(reinterpret_cast<void**>(src_sp), reinterpret_cast<void**>(dest_sp));
}

```

```
.globl ctx_swap
.type  ctx_swap, @function
ctx_swap:
      pushq %rdi
      pushq %r12
      pushq %r13
      pushq %r14
      pushq %r15
      pushq %rbx
      pushq %rbp
      movq %rsp, (%rdi)

      movq (%rsi), %rsp
      popq %rbp
      popq %rbx
      popq %r15
      popq %r14
      popq %r13
      popq %r12
      popq %rdi
      ret
```

### croutine

```
using RoutineFunc = std::function<void()>;
using Duration = std::chrono::microseconds;

enum class RoutineState { READY, FINISHED, SLEEP, IO_WAIT, DATA_WAIT };

class CRoutine {
 public:
  explicit CRoutine(const RoutineFunc &func);
  virtual ~CRoutine();

  // static interfaces
  static void Yield();
  static void Yield(const RoutineState &state);
  static void SetMainContext(const std::shared_ptr<RoutineContext> &context);
  static CRoutine *GetCurrentRoutine();
  static char **GetMainStack();

  // public interfaces
  bool Acquire();
  void Release();

  // It is caller's responsibility to check if state_ is valid before calling
  // SetUpdateFlag().
  void SetUpdateFlag();

  // acquire && release should be called before Resume
  // when work-steal like mechanism used
  RoutineState Resume();
  RoutineState UpdateState();
  RoutineContext *GetContext();
  char **GetStack();

  void Run();
  void Stop();
  void Wake();
  void HangUp();
  void Sleep(const Duration &sleep_duration);

  // getter and setter
  RoutineState state() const;
  void set_state(const RoutineState &state);

  uint64_t id() const;
  void set_id(uint64_t id);

  const std::string &name() const;
  void set_name(const std::string &name);

  int processor_id() const;
  void set_processor_id(int processor_id);

  uint32_t priority() const;
  void set_priority(uint32_t priority);

  std::chrono::steady_clock::time_point wake_time() const;

  void set_group_name(const std::string &group_name) {
    group_name_ = group_name;
  }

  const std::string &group_name() { return group_name_; }

 private:
  CRoutine(CRoutine &) = delete;
  CRoutine &operator=(CRoutine &) = delete;

  std::string name_;
  std::chrono::steady_clock::time_point wake_time_ =
      std::chrono::steady_clock::now();

  RoutineFunc func_;
  RoutineState state_;

  std::shared_ptr<RoutineContext> context_;

  std::atomic_flag lock_ = ATOMIC_FLAG_INIT;
  std::atomic_flag updated_ = ATOMIC_FLAG_INIT;

  bool force_stop_ = false;

  int processor_id_ = -1;
  uint32_t priority_ = 0;
  uint64_t id_ = 0;

  std::string group_name_;

  static thread_local CRoutine *current_routine_;
  static thread_local char *main_stack_;
};
```

## data

### cache_buffer

```
template <typename T>
class CacheBuffer {
 public:
  using value_type = T;
  using size_type = std::size_t;
  using FusionCallback = std::function<void(const T&)>;

  explicit CacheBuffer(uint64_t size) {
    capacity_ = size + 1;
    buffer_.resize(capacity_);
  }

  CacheBuffer(const CacheBuffer& rhs) {
    std::lock_guard<std::mutex> lg(rhs.mutex_);
    head_ = rhs.head_;
    tail_ = rhs.tail_;
    buffer_ = rhs.buffer_;
    capacity_ = rhs.capacity_;
    fusion_callback_ = rhs.fusion_callback_;
  }

  T& operator[](const uint64_t& pos) { return buffer_[GetIndex(pos)]; }
  const T& at(const uint64_t& pos) const { return buffer_[GetIndex(pos)]; }

  uint64_t Head() const { return head_ + 1; }
  uint64_t Tail() const { return tail_; }
  uint64_t Size() const { return tail_ - head_; }

  const T& Front() const { return buffer_[GetIndex(head_ + 1)]; }
  const T& Back() const { return buffer_[GetIndex(tail_)]; }

  bool Empty() const { return tail_ == 0; }
  bool Full() const { return capacity_ - 1 == tail_ - head_; }
  uint64_t Capacity() const { return capacity_; }

  void SetFusionCallback(const FusionCallback& callback) {
    fusion_callback_ = callback;
  }

  void Fill(const T& value) {
    if (fusion_callback_) {
      fusion_callback_(value);
    } else {
      if (Full()) {
        buffer_[GetIndex(head_)] = value;
        ++head_;
        ++tail_;
      } else {
        buffer_[GetIndex(tail_ + 1)] = value;
        ++tail_;
      }
    }
  }

  std::mutex& Mutex() { return mutex_; }

 private:
  CacheBuffer& operator=(const CacheBuffer& other) = delete;
  uint64_t GetIndex(const uint64_t& pos) const { return pos % capacity_; }

  uint64_t head_ = 0;
  uint64_t tail_ = 0;
  uint64_t capacity_ = 0;
  std::vector<T> buffer_;
  mutable std::mutex mutex_;
  FusionCallback fusion_callback_;
};
```

### channel_buffer

```
using apollo::cyber::common::GlobalData;

template <typename T>
class ChannelBuffer {
 public:
  using BufferType = CacheBuffer<std::shared_ptr<T>>;
  ChannelBuffer(uint64_t channel_id, BufferType* buffer)
      : channel_id_(channel_id), buffer_(buffer) {}

  bool Fetch(uint64_t* index, std::shared_ptr<T>& m);  // NOLINT

  bool Latest(std::shared_ptr<T>& m);  // NOLINT

  bool FetchMulti(uint64_t fetch_size, std::vector<std::shared_ptr<T>>* vec);

  uint64_t channel_id() const { return channel_id_; }
  std::shared_ptr<BufferType> Buffer() const { return buffer_; }

 private:
  uint64_t channel_id_;
  std::shared_ptr<BufferType> buffer_;
};

template <typename T>
bool ChannelBuffer<T>::Fetch(uint64_t* index,
                             std::shared_ptr<T>& m) {  // NOLINT
  std::lock_guard<std::mutex> lock(buffer_->Mutex());
  if (buffer_->Empty()) {
    return false;
  }

  if (*index == 0) {
    *index = buffer_->Tail();
  } else if (*index == buffer_->Tail() + 1) {
    return false;
  } else if (*index < buffer_->Head()) {
    auto interval = buffer_->Tail() - *index;
    AWARN << "channel[" << GlobalData::GetChannelById(channel_id_) << "] "
          << "read buffer overflow, drop_message[" << interval << "] pre_index["
          << *index << "] current_index[" << buffer_->Tail() << "] ";
    *index = buffer_->Tail();
  }
  m = buffer_->at(*index);
  return true;
}

template <typename T>
bool ChannelBuffer<T>::Latest(std::shared_ptr<T>& m) {  // NOLINT
  std::lock_guard<std::mutex> lock(buffer_->Mutex());
  if (buffer_->Empty()) {
    return false;
  }

  m = buffer_->Back();
  return true;
}

template <typename T>
bool ChannelBuffer<T>::FetchMulti(uint64_t fetch_size,
                                  std::vector<std::shared_ptr<T>>* vec) {
  std::lock_guard<std::mutex> lock(buffer_->Mutex());
  if (buffer_->Empty()) {
    return false;
  }

  auto num = std::min(buffer_->Size(), fetch_size);
  vec->reserve(num);
  for (auto index = buffer_->Tail() - num + 1; index <= buffer_->Tail();
       ++index) {
    vec->emplace_back(buffer_->at(index));
  }
  return true;
}
```

## doxy-docs


## event 


## io


## logger


## mainboard


## message

### message_traits

```
template <typename T>
class HasSerializer
    static constexpr bool value =
      HasSerializeToString<T>::value && HasParseFromString<T>::value &&
      HasSerializeToArray<T>::value && HasParseFromArray<T>::value;
//判断T中是否有SerializeToString、ParseFromString、SerializeToArray、ParseFromArray成员函数

std::string MessageType(const T& message)
std::string MessageType()
//模板函数，两个重载版本，
//1.HasType并且TypeNmae是静态成员函数，T::TypeName() 
//2.HasType并且T不是proto Message类或其派生类，typeid(T).name();

SetTypeName

ByteSize

FullByteSize

ParseFromArray

ParseFromString

ParseFromHC

SerializeToArray

SerializeToString

SerializeToHC

GetDescriptorString

GetFullName

GetMessageName
```

## node

### writer_base

```
class WriterBase {
 public:
  explicit WriterBase(const proto::RoleAttributes& role_attr)
      : role_attr_(role_attr), init_(false) {}
  virtual ~WriterBase() {}

  virtual bool Init() = 0;

  virtual void Shutdown() = 0;

  virtual bool HasReader() { return false; }

  virtual void GetReaders(std::vector<proto::RoleAttributes>* readers) {}

  const std::string& GetChannelName() const {
    return role_attr_.channel_name();
  }

  bool IsInit() const {
    std::lock_guard<std::mutex> g(lock_);
    return init_;
  }

 protected:
  proto::RoleAttributes role_attr_;
  mutable std::mutex lock_;
  bool init_;
};
```

### writer_base


```
template <typename MessageT>
class Writer : public WriterBase {
 public:
  using TransmitterPtr = std::shared_ptr<transport::Transmitter<MessageT>>;
  using ChangeConnection =
      typename service_discovery::Manager::ChangeConnection;

  explicit Writer(const proto::RoleAttributes& role_attr);
  virtual ~Writer();

  bool Init() override;

  void Shutdown() override;

  virtual bool Write(const MessageT& msg);

  virtual bool Write(const std::shared_ptr<MessageT>& msg_ptr);

  bool HasReader() override;

  void GetReaders(std::vector<proto::RoleAttributes>* readers) override;

 private:
  void JoinTheTopology();
  void LeaveTheTopology();
  void OnChannelChange(const proto::ChangeMsg& change_msg);

  TransmitterPtr transmitter_;

  ChangeConnection change_conn_;
  service_discovery::ChannelManagerPtr channel_manager_;
};
```

### reader_base


## parameter


## proto


## python


## record


## scheduler


## service


## service_discovery


## sysmo


## task


## time


## tools


## transport

### common

#### identity

//uuid及其hash值
```
class Identity {
 public:
  explicit Identity(bool need_generate = true);
  Identity(const Identity& another);
  virtual ~Identity();

  Identity& operator=(const Identity& another);
  bool operator==(const Identity& another) const;
  bool operator!=(const Identity& another) const;

  std::string ToString() const;
  size_t Length() const;
  uint64_t HashValue() const;

  const char* data() const { return data_; }
  void set_data(const char* data) {
    if (data == nullptr) {
      return;
    }
    std::memcpy(data_, data, sizeof(data_));
    Update();
  }

 private:
  void Update();

  char data_[ID_SIZE];
  uint64_t hash_value_;
};
```

#### endpoint

```
class Endpoint              //从GlobalData中读取role信息
    bool enabled_;
    Identity id_;
    RoleAttributes attr_;   //proto，host、node、channel、socket等相关信息

class Endpoint;
using EndpointPtr = std::shared_ptr<Endpoint>;

using proto::RoleAttributes;

class Endpoint {
 public:
  explicit Endpoint(const RoleAttributes& attr);
  virtual ~Endpoint();

  const Identity& id() const { return id_; }
  const RoleAttributes& attributes() const { return attr_; }

 protected:
  bool enabled_;
  Identity id_;
  RoleAttributes attr_;
};
```

### message

```
struct HistoryAttributes
    proto::QosHistoryPolicy history_policy;     //proto QosProfile,QosHistoryPolicy默认1即HISTORY_KEEP_LAST
    uint32_t depth;


template <typename MessageT>
class History               //GlobalData读取config

    struct CachedMessage    //message智能指针和messageinfo
        MessagePtr msg;
        MessageInfo msg_info;

    bool enabled_;
    uint32_t depth_;
    uint32_t max_depth_;
    std::list<CachedMessage> msgs_;
    mutable std::mutex msgs_mutex_;


```


## others

binary.h&&binary.cc

NOLINT          //静态检查不报错
```
namespace {
std::mutex m;
std::string binary_name; // NOLINT
}  // namespace

namespace apollo {
namespace cyber {
namespace binary {

std::string GetName() {
  std::lock_guard<std::mutex> lock(m);
  return binary_name;
}
void SetName(const std::string& name) {
  std::lock_guard<std::mutex> lock(m);
  binary_name = name;
}

}  // namespace binary
}  // namespace cyber
} 
```

state.h&&state.cc

```
.h
enum State : std::uint8_t {
  STATE_UNINITIALIZED = 0,
  STATE_INITIALIZED,
  STATE_SHUTTING_DOWN,
  STATE_SHUTDOWN,
};
State GetState();
void SetState(const State& state);

inline bool OK() { return GetState() == STATE_INITIALIZED; }

inline bool IsShutdown() {
  return GetState() == STATE_SHUTTING_DOWN || GetState() == STATE_SHUTDOWN;
}

inline void WaitForShutdown() {
  while (!IsShutdown()) {
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
  }
}

inline void AsyncShutdown() {
  pid_t pid = getpid();
  if (kill(pid, SIGINT) != 0) {
    AERROR << strerror(errno);
  }
}
.cc
namespace {
std::atomic<State> g_cyber_state;
}

State GetState() { return g_cyber_state.load(); }

void SetState(const State& state) { g_cyber_state.store(state); }

```

cyber.h&&cyber.cc

创建节点
```
namespace apollo {
namespace cyber {

using apollo::cyber::common::GlobalData;
using apollo::cyber::proto::RunMode;

std::unique_ptr<Node> CreateNode(const std::string& node_name,
                                 const std::string& name_space) {
  bool is_reality_mode = GlobalData::Instance()->IsRealityMode();
  if (is_reality_mode && !OK()) {
    // add some hint log
    AERROR << "please initialize cyber firstly.";
    return nullptr;
  }
  return std::unique_ptr<Node>(new Node(node_name, name_space));
}

}  // namespace cyber
}  // namespace apollo
```

init.h&&init.cc


```
using apollo::cyber::scheduler::Scheduler;
using apollo::cyber::service_discovery::TopologyManager;
namespace {

const std::string& kClockChannel = "/clock";
const std::string& kClockNode = "clock";

bool g_atexit_registered = false;
std::mutex g_mutex;
std::unique_ptr<Node> clock_node;

logger::AsyncLogger* async_logger = nullptr;

void InitLogger(const char* binary_name) {
  const char* slash = strrchr(binary_name, '/');
  if (slash) {
    ::apollo::cyber::binary::SetName(slash + 1);
  } else {
    ::apollo::cyber::binary::SetName(binary_name);
  }

  // Init glog
  google::InitGoogleLogging(binary_name);
  google::SetLogDestination(google::ERROR, "");
  google::SetLogDestination(google::WARNING, "");
  google::SetLogDestination(google::FATAL, "");

  // Init async logger
  async_logger = new ::apollo::cyber::logger::AsyncLogger(
      google::base::GetLogger(FLAGS_minloglevel));
  google::base::SetLogger(FLAGS_minloglevel, async_logger);
  async_logger->Start();
}

void StopLogger() { delete async_logger; }

}  // namespace
bool Init(const char* binary_name);
//初始状态未初始化，InitLogger即设置glog,开启异步日志async_logger
void Clear();
```