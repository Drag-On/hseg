//
// Created by jan on 25.08.16.
//

#include <iostream>
#include <iomanip>
#include "Timer.h"

Timer::Timer(bool run)
{
    reset();
    if(run)
        start();
    else
        m_paused = true;
}

void Timer::reset(bool restart)
{
    m_duration = nanoseconds::zero();
    if(restart)
        start();
}

void Timer::pause()
{
    m_duration += std::chrono::high_resolution_clock::now() - m_start;
    m_paused = true;
}

void Timer::start()
{
    m_paused = false;
    m_start = std::chrono::high_resolution_clock::now();
}

namespace Profiler
{
    ProfilerLog g_pProfiler_log;

    FunctionTracer::FunctionTracer(std::string const& funcName) noexcept
            : m_timer(true),
              m_funcName(funcName)
    {
    }

    FunctionTracer::~FunctionTracer() noexcept
    {
        Profiler::g_pProfiler_log.log(m_funcName, m_timer.elapsed());
    }

    ProfilerLog::~ProfilerLog() noexcept
    {
        std::cout << std::endl << "== PROFILING RESULTS ==" << std::endl;
        for(auto const& r : m_records)
        {
            std::cout << std::setw(32) << r.first << ": "
                      << std::setw(8) << r.second.m_totalTime << " TOTAL, "
                      << std::setw(8) << (r.second.m_totalTime / r.second.m_numCalled).count()  << "ms AVERAGE, "
                      << std::setw(8) << r.second.m_numCalled << " CALLS" << std::endl;
        }
    }

    void ProfilerLog::log(std::string const& funcName, std::chrono::milliseconds const& time) noexcept
    {
        if(m_records.count(funcName) > 0)
        {
            Record& r = m_records.at(funcName);
            r.m_numCalled++;
            r.m_totalTime += time;
        }
        else
        {
            Record r;
            r.m_numCalled = 1;
            r.m_totalTime = time;
            m_records.emplace(funcName, r);
        }
    }
}
