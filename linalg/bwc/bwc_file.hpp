#ifndef CADO_LINALG_BWC_BWC_FILE_HPP
#define CADO_LINALG_BWC_BWC_FILE_HPP

#include <condition_variable>
#include <deque>
#include <functional>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

/* a file to write, gathered at the leader */
struct bwc_file {
    std::string name;
    std::vector<char> data;
};

/* write name.tmp and rename it to name; print and return 0 on failure */
int bwc_write_file(bwc_file const & f);

/* write the name.tmp of every file, one thread per file, then rename
 * them all, so that the set of files of one checkpoint appears at once;
 * abort on failure, so a failed save stops the computation instead of
 * passing for a checkpoint */
void bwc_write_files(std::vector<bwc_file> const & files);

/* jobs run one after the other in a background thread; run waits
 * while a job is pending behind the running one, so that one slow
 * checkpoint write overlaps the next interval and only two in a row
 * stop the iterations */
struct bwc_background {
    std::mutex m;
    std::condition_variable cv;
    std::deque<std::function<void()>> jobs;
    bool busy = false;
    bool stop = false;
    std::thread t { [this] { work(); } };
    ~bwc_background();
    void run(std::function<void()> job);
    void wait();
    void work();
};

#endif	/* CADO_LINALG_BWC_BWC_FILE_HPP */
