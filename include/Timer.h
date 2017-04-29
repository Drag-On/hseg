//
// Created by jan on 25.08.16.
//

#ifndef HSEG_TIMER_H
#define HSEG_TIMER_H

#include <chrono>
#include <ostream>
#include <map>

/**
 * Can be used to measure time intervals
 */
class Timer
{
public:
    using nanoseconds = std::chrono::nanoseconds;
    using microseconds = std::chrono::microseconds;
    using milliseconds = std::chrono::milliseconds;
    using seconds = std::chrono::seconds;
    using minutes = std::chrono::minutes;
    using hours = std::chrono::hours;

    /**
     * Constructor
     * @param run Indicated whether or not the timer should be started immediately
     */
    explicit Timer(bool run = false);

    /**
     * Reset the timer
     * @param restart Indicates whether or not the timer should be started immediately
     */
    void reset(bool restart = false);

    /**
     * Pause the timer but keep the current elapsed time
     */
    void pause();

    /**
     * Starts the timer
     */
    void start();

    /**
     * @return The elapsed time since the timer has last been reset
     * @tparam T Time unit
     */
    template<typename T = milliseconds>
    T elapsed() const;

private:
    std::chrono::high_resolution_clock::time_point m_start;
    nanoseconds m_duration;
    bool m_paused;
};

template<typename T, typename Traits>
std::basic_ostream<T, Traits>& operator<<(std::basic_ostream<T, Traits>& out, Timer::nanoseconds const& ns)
{
    return out << ns.count() << "ns";
}

template<typename T, typename Traits>
std::basic_ostream<T, Traits>& operator<<(std::basic_ostream<T, Traits>& out, Timer::microseconds const& ms)
{
    return out << ms.count() << "us";
}

template<typename T, typename Traits>
std::basic_ostream<T, Traits>& operator<<(std::basic_ostream<T, Traits>& out, Timer::milliseconds const& ms)
{
    return out << ms.count() << "ms";
}

template<typename T, typename Traits>
std::basic_ostream<T, Traits>& operator<<(std::basic_ostream<T, Traits>& out, Timer::seconds const& s)
{
    return out << s.count() << "s";
}

template<typename T, typename Traits>
std::basic_ostream<T, Traits>& operator<<(std::basic_ostream<T, Traits>& out, Timer::minutes const& m)
{
    return out << m.count() << "min";
}

template<typename T, typename Traits>
std::basic_ostream<T, Traits>& operator<<(std::basic_ostream<T, Traits>& out, Timer::hours const& h)
{
    return out << h.count() << "h";
}

template<typename T>
T Timer::elapsed() const
{
    auto duration = m_duration;
    if (!m_paused)
        duration += std::chrono::high_resolution_clock::now() - m_start;
    return std::chrono::duration_cast<T>(duration);
}

namespace Profiler
{
    class FunctionTracer
    {
    private:
        Timer m_timer;
        std::string m_funcName;
    public:
        explicit FunctionTracer(std::string const& funcName) noexcept;
        ~FunctionTracer() noexcept;
    };

    class ProfilerLog
    {
    private:
        struct Record
        {
            size_t m_numCalled = 0;
            std::chrono::milliseconds m_totalTime{0};
        };
        std::map<std::string,Record> m_records;
    public:
        ~ProfilerLog() noexcept;
        void log(std::string const& funcName, std::chrono::milliseconds const& time) noexcept;
    };

    extern ProfilerLog g_profiler_log;
}

#if defined(__PRETTY_FUNCTION__)
#define CUR_FUNCTION __PRETTY_FUNCTION__
#elif defined(__FUNCTION__)
#define CUR_FUNCTION __FUNCTION__
#else
#define CUR_FUNCTION __func__
#endif

#if defined(USE_PROFILER)
#define PROFILE_THIS volatile Profiler::FunctionTracer profiler_tracer_(CUR_FUNCTION);
#else
#define PROFILE_THIS
#endif


#endif //HSEG_TIMER_H
