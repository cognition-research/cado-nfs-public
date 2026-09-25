#include "cado.h" // IWYU pragma: keep

#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>

#include "fmt/base.h"

#include "bwc_file.hpp"
#include "misc.h"
#include "timing.h"

static int write_tmp(bwc_file const & f)
{
    std::string const tmp = f.name + ".tmp";
    FILE * fp = fopen(tmp.c_str(), "wb");
    if (!fp) {
        fmt::print(stderr, "ERROR: failed to save {}: {}\n", f.name, strerror(errno));
        return 0;
    }
    size_t const n = fwrite(f.data.data(), 1, f.data.size(), fp);
    int ok = n == f.data.size();
    if (!ok)
        fmt::print(stderr, "ERROR: failed to save {}: short write, {}\n", f.name, strerror(errno));
    if (fclose(fp) != 0) {
        fmt::print(stderr, "ERROR: failed to save {}: failed fclose, {}\n", f.name, strerror(errno));
        ok = 0;
    }
    return ok;
}

static int publish(bwc_file const & f)
{
    std::string const tmp = f.name + ".tmp";
    if (rename(tmp.c_str(), f.name.c_str()) != 0) {
        fmt::print(stderr, "ERROR: failed to save {}: failed rename, {}\n", f.name, strerror(errno));
        return 0;
    }
    return 1;
}

int bwc_write_file(bwc_file const & f)
{
    return write_tmp(f) && publish(f);
}

void bwc_write_files(std::vector<bwc_file> const & files)
{
    double tt = -wct_seconds();
    std::vector<int> ok(files.size());
    std::vector<std::thread> writers;
    for(size_t i = 0 ; i < files.size() ; i++)
        writers.emplace_back([&, i] { ok[i] = write_tmp(files[i]); });
    for(auto & t : writers)
        t.join();
    for(size_t i = 0 ; i < files.size() ; i++) {
        if (ok[i])
            continue;
        fmt::print(stderr, "FATAL: could not save {}\n", files[i].name);
        abort();
    }
    for(auto const & f : files) {
        if (publish(f))
            continue;
        fmt::print(stderr, "FATAL: could not save {}\n", f.name);
        abort();
    }
    tt += wct_seconds();
    size_t total = 0;
    for(auto const & f : files) {
        total += f.data.size();
        fmt::print("Saved {} [{}]\n", f.name, size_disp(f.data.size()));
    }
    fmt::print("Saved {} files [{} in {:.2f}, {}/s]\n", files.size(),
            size_disp(total), tt, size_disp(total / tt));
}

void bwc_background::run(std::function<void()> job)
{
    std::unique_lock<std::mutex> lock(m);
    cv.wait(lock, [this] { return jobs.empty(); });
    jobs.push_back(std::move(job));
    cv.notify_all();
}

void bwc_background::wait()
{
    std::unique_lock<std::mutex> lock(m);
    cv.wait(lock, [this] { return jobs.empty() && !busy; });
}

void bwc_background::work()
{
    std::unique_lock<std::mutex> lock(m);
    for(;;) {
        cv.wait(lock, [this] { return stop || !jobs.empty(); });
        if (jobs.empty())
            return;
        std::function<void()> job = std::move(jobs.front());
        jobs.pop_front();
        busy = true;
        lock.unlock();
        job();
        lock.lock();
        busy = false;
        cv.notify_all();
    }
}

bwc_background::~bwc_background()
{
    wait();
    { std::lock_guard<std::mutex> const lock(m); stop = true; }
    cv.notify_all();
    t.join();
}
