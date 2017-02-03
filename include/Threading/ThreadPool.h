//////////////////////////////////////////////////////////////////////
/// Dragon Blaze Game Library
///
/// Copyright (c) 2015 by Jan Moeller
///
/// This software is provided "as-is" and does not claim to be
/// complete or free of bugs in any way. It should work, but
/// it might also begin to hurt your kittens.
//////////////////////////////////////////////////////////////////////

#ifndef DBGL_THREADPOOL_H
#define DBGL_THREADPOOL_H

#include <future>
#include <thread>
#include <functional>
#include <mutex>
#include <vector>
#include <deque>

/**
 * @brief A thread pool which works off its jobs on a first-come-first-served basis
 * @details If the thread pool is getting destructed before all the jobs are done, then it will finish all the
 *          computations that have already been started, and then skips anything left in the queue. Thus it's the
 *          programmers responsibility to keep the thread pool alive as long as needed.
 */
class ThreadPool
{
private:
    /**
     * @brief Interface class for jobs
     */
    class IJob
    {
    public:
        /**
         * @brief Destructor
         */
        virtual ~IJob() = default;

        /**
         * @brief Executes the job
         */
        virtual void execute() = 0;
    };

    /**
     * @brief Actual job implementation
     */
    template<typename R>
    class Job : public IJob
    {
    private:
        std::packaged_task<R()> m_task;
    public:
        /**
         * @brief Constructs a job from a task
         * @param fun Function to execute
         */
        template<typename T>
        Job(T fun);

        /**
         * @copydoc IJob::execute()
         */
        virtual void execute();

        /**
         * @brief Retrieves the future object
         */
        auto getFuture() -> decltype(m_task.get_future());
    };

public:
    /**
     * @brief Creates a thread pool with a number of threads
     * @param threads Amount of threads to generate
     */
    explicit ThreadPool(unsigned int threads);

    /**
     * @brief Destructor
     */
    ~ThreadPool();

    /**
     * @brief Enqueue a job to the thread pool
     * @details As soon as there is a thread available, the job will be processed
     * @param fun Function executing the job
     * @param args Arguments to pass to \p fun
     * @returns A std::future which can be used to access the computation result
     */
    template<typename Fun, typename... Args>
    auto enqueue(Fun fun, Args... args) -> std::future<decltype(fun(args...))>;

    /**
     * @brief Retrieves the amount of threads within this pool
     * @return Amount of threads within the pool
     */
    unsigned int size() const;

    /**
     * @return The amount of currently queued jobs (disregarding those currently served by a thread)
     */
    size_t queued() const;

private:
    void runThread();

    std::vector<std::thread> m_threads;
    std::deque<IJob*> m_jobs;
    std::mutex m_internalMutex;
    bool m_shutdown = false;
};

#include "ThreadPool.inl"

#endif //DBGL_THREADPOOL_H
