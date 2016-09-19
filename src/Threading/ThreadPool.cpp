//////////////////////////////////////////////////////////////////////
/// Dragon Blaze Game Library
///
/// Copyright (c) 2015 by Jan Moeller
///
/// This software is provided "as-is" and does not claim to be
/// complete or free of bugs in any way. It should work, but
/// it might also begin to hurt your kittens.
//////////////////////////////////////////////////////////////////////

#include "Threading/ThreadPool.h"

ThreadPool::ThreadPool(unsigned int threads)
{
    m_internalMutex.lock();
    m_threads.reserve(threads);
    for (unsigned int i = 0; i < threads; i++)
        m_threads.emplace_back(std::move(std::bind(&ThreadPool::runThread, this)));
    m_internalMutex.unlock();
}

ThreadPool::~ThreadPool()
{
    m_internalMutex.lock();
    m_shutdown = true;
    m_internalMutex.unlock();
    for (auto& t : m_threads)
        t.join();
    for (auto j : m_jobs)
        delete j;
}

void ThreadPool::runThread()
{
    while (true)
    {
        m_internalMutex.lock();
        if (m_shutdown)
        {
            m_internalMutex.unlock();
            return;
        }
        if (m_jobs.empty() && !m_shutdown)
        {
            m_internalMutex.unlock();
            std::this_thread::yield();
            continue;
        }
        else if (!m_jobs.empty())
        {
            auto job = m_jobs.front();
            m_jobs.pop_front();
            m_internalMutex.unlock();
            job->execute();
            delete job;
            continue;
        }
    }
}

unsigned int ThreadPool::size()
{
    return m_threads.size();
}
