//
// Created by jan on 25.08.16.
//

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
