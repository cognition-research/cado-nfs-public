#ifndef CADO_ASYNC_HPP
#define CADO_ASYNC_HPP

#include <cstddef>

#include "macros.h"
#include "parallelizing_info.hpp"
#include "timing.h"

/* How long do we typically accept to wait for asynchronous output (=
 * signals) to be checked ? */
#define PREFERRED_ASYNC_LAG     15.0

/* This structure **MUST** be made only of doubles, or disaster will
 * occur */
struct timing_interval_data_s {
    double job[2];  // user, system
    double thread[2];       // user, system
    double wct;
};
typedef struct timing_interval_data_s timing_interval_data[1];

struct timing_data {
    int go_mark;
    unsigned int begin_mark;
    unsigned int end_mark;
    int last_print;
    int next_print;
    int which;
    int ntimers;
    char ** names;
    size_t * items;
    timing_interval_data * since_last_reset;
    timing_interval_data * since_beginning;
    /* values stored in the timing_interval_data structures correspond to
     * time interval values, except for the [which] structure, where it
     * is (the opposite of) the time since when we've been measuring on
     * its account. Switching from a timer to another then means getting
     * a clock tick, adding it to counter [which], and subtracting it to
     * another (which will then become [which]).
     */
};

void timing_init(struct timing_data * t, int n, int start, int end);
void timing_set_timer_name(struct timing_data * t, int i, const char *, ...)
    ATTR_PRINTF(3, 4);
void timing_set_timer_items(struct timing_data * t, int i, size_t count);
void timing_clear(struct timing_data * t);
void timing_next_timer(struct timing_data * t);

void timing_check(parallelizing_info & pi, struct timing_data * t, int iter, int print);
void timing_update_ticks(struct timing_data * t, int iter);
void timing_disp_collective_oneline(parallelizing_info & pi, struct timing_data * timing, int iter, int print, const char * stage);
void timing_final_tally(parallelizing_info & pi, struct timing_data * timing, int print, const char * stage);

/* lap() is the wall time since the construction or the previous lap */
struct epoch_timer {
    double t = wct_seconds();
    double lap() {
        double const t0 = t;
        t = wct_seconds();
        return t - t0;
    }
};

/* One line per epoch of n iterations ending at iter: the wall time per
 * iteration of the iterations alone, and the wall time of the rest of
 * the epoch (checkpoint loads and saves, checks), so that a rate read
 * from the log does not carry the file system. */
void epoch_report(int print, const char * stage, unsigned int iter, unsigned int n, double iter_s, double other_s);

#endif	/* CADO_ASYNC_HPP */
