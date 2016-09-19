//////////////////////////////////////////////////////////////////////
/// Dragon Blaze Game Library
///
/// Copyright (c) 2015 by Jan Moeller
///
/// This software is provided "as-is" and does not claim to be
/// complete or free of bugs in any way. It should work, but
/// it might also begin to hurt your kittens.
//////////////////////////////////////////////////////////////////////

template<typename R>
template<typename T>
ThreadPool::Job<R>::Job(T fun) : m_task{fun}
{
}

template<typename R>
void ThreadPool::Job<R>::execute()
{
    m_task();
}

template<typename R>
auto ThreadPool::Job<R>::getFuture() -> decltype(m_task.get_future())
{
    return m_task.get_future();
}

template<typename Fun, typename... Args>
auto ThreadPool::enqueue(Fun fun, Args... args) -> std::future<decltype(fun(args...))>
{
    using Result = decltype(fun(args...));

    Job <Result>* job = new Job <Result>{std::move(std::bind(fun, args...))};

    m_internalMutex.lock();
    m_jobs.push_back(job);
    auto future = job->getFuture();
    m_internalMutex.unlock();

    return future;
}
