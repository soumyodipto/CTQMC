// Lightweight assert-based tests for ctqmc_1d.cpp (no external test framework).
// Build/run with: make test
#include <cmath>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <armadillo>
#include "../ctqmc_1d.h"

static int g_failures = 0;
static int g_checks = 0;

#define CHECK(cond)                                                          \
    do {                                                                     \
        ++g_checks;                                                          \
        if (!(cond)) {                                                       \
            ++g_failures;                                                    \
            std::cerr << __FILE__ << ":" << __LINE__ << ": CHECK failed: "  \
                      << #cond << std::endl;                                 \
        }                                                                    \
    } while (0)

// Runs run_simulation() with std::cout redirected; returns captured output.
static std::string capture_run(int* ret)
{
    std::ostringstream buf;
    std::streambuf* old = std::cout.rdbuf(buf.rdbuf());
    *ret = run_simulation();
    std::cout.rdbuf(old);
    return buf.str();
}

static bool contains_line_starting_with(const std::string& out, const std::string& prefix)
{
    std::istringstream in(out);
    std::string line;
    while (std::getline(in, line))
        if (line.compare(0, prefix.size(), prefix) == 0)
            return true;
    return false;
}

// Element-wise equality that treats NaN == NaN (CTQMC observables can be NaN)
static bool same_values(const std::vector<double>& a, const std::vector<double>& b)
{
    if (a.size() != b.size())
        return false;
    for (size_t i = 0; i < a.size(); ++i)
        if (!(a[i] == b[i] || (std::isnan(a[i]) && std::isnan(b[i]))))
            return false;
    return true;
}

static std::string strip_time_line(const std::string& out)
{
    std::istringstream in(out);
    std::ostringstream res;
    std::string line;
    while (std::getline(in, line))
        if (line.compare(0, 6, "time: ") != 0)
            res << line << "\n";
    return res.str();
}

// Temporarily shrink the Markov chain so tests run quickly.
struct SmallChain {
    int steps, equil, interval;
    SmallChain() : steps(numMCsteps), equil(numEquil), interval(numObsInterval)
    {
        numMCsteps = 100;
        numEquil = 20;
        numObsInterval = 5;
    }
    ~SmallChain()
    {
        numMCsteps = steps;
        numEquil = equil;
        numObsInterval = interval;
        set_seed_random();
    }
};

static void test_run_simulation_output()
{
    SmallChain small;
    set_seed(12345);
    int ret = -1;
    std::string out = capture_run(&ret);
    CHECK(ret == 0);
    CHECK(contains_line_starting_with(out, "Temperature = "));
    CHECK(contains_line_starting_with(out, "Lambda = "));
    CHECK(contains_line_starting_with(out, "time: "));

    std::ostringstream expT, expL;
    expT << "Temperature = " << T << "\n";
    expL << "Lambda = " << lam << "\n";
    CHECK(out.find(expT.str()) == 0);
    CHECK(out.find(expL.str()) == expT.str().size());
}

static void test_deterministic_with_fixed_seed()
{
    SmallChain small;
    // Observations are taken at steps > numEquil with step % numObsInterval == 0
    const size_t expected_obs = 15;  // steps 25, 30, ..., 95

    set_seed(42);
    int ret1 = -1;
    std::string out1 = capture_run(&ret1);
    std::vector<double> num1 = last_num_trace;
    std::vector<int> ord1 = last_order_trace;

    set_seed(42);
    int ret2 = -1;
    std::string out2 = capture_run(&ret2);
    std::vector<double> num2 = last_num_trace;
    std::vector<int> ord2 = last_order_trace;

    CHECK(ret1 == 0);
    CHECK(ret2 == 0);
    CHECK(strip_time_line(out1) == strip_time_line(out2));
    CHECK(num1.size() == expected_obs);
    CHECK(ord1.size() == expected_obs);
    CHECK(same_values(num1, num2));
    CHECK(ord1 == ord2);
    for (size_t i = 0; i < ord1.size(); ++i)
        CHECK(ord1[i] >= 0);

    set_seed(7);
    int ret3 = -1;
    capture_run(&ret3);
    CHECK(ret3 == 0);
    CHECK(last_order_trace != ord1);
}

static void test_mc_move_decision_accepts_ratio_ge_one()
{
    set_seed(1);
    // insert: R = V0 * w_new * beta / w_old / order_new
    CHECK(MCMoveDecision(0, 1, 1.0, 1.0, "insert"));
    CHECK(MCMoveDecision(1, 2, 1.0, 1.0 / (V0 * beta / 2.0), "insert"));  // R == 1

    // remove: R = w_new * order_old / (V0 * w_old * beta)
    CHECK(MCMoveDecision(1, 0, 1.0, 1000.0 * V0 * beta, "remove"));
    CHECK(MCMoveDecision(2, 1, 1.0, V0 * beta / 2.0, "remove"));  // R == 1

    // stay: R = w_new / w_old
    CHECK(MCMoveDecision(3, 3, 1.0, 2.0, "stay"));
    CHECK(MCMoveDecision(3, 3, 1.5, 1.5, "stay"));  // R == 1

    // Sanity: ratio of 0 is never accepted, unknown move never accepted
    for (int i = 0; i < 100; ++i) {
        CHECK(!MCMoveDecision(3, 3, 1.0, 0.0, "stay"));
        CHECK(!MCMoveDecision(3, 3, 1.0, 2.0, "bogus"));
    }
}

static void test_pack_new_features()
{
    set_seed(2024);
    arma::vec taus2 = {0.8 * beta, 0.3 * beta};  // descending order, order 2

    auto ins = PackNewFeatures(taus2, 2, "insert");
    CHECK(std::get<0>(ins) == 3);
    CHECK(std::get<1>(ins).n_elem == 3);
    CHECK(std::get<5>(ins).n_rows == 3 && std::get<5>(ins).n_cols == 3);
    CHECK(std::isfinite(std::get<6>(ins)) && std::get<6>(ins) > 0.0);

    auto rem = PackNewFeatures(taus2, 2, "remove");
    CHECK(std::get<0>(rem) == 1);
    CHECK(std::get<1>(rem).n_elem == 1);
    CHECK(std::isfinite(std::get<6>(rem)) && std::get<6>(rem) > 0.0);

    auto stay = PackNewFeatures(taus2, 2, "stay");
    CHECK(std::get<0>(stay) == 2);
    CHECK(std::get<1>(stay).n_elem == 2);
    CHECK(std::isfinite(std::get<6>(stay)) && std::get<6>(stay) > 0.0);

    // Removing the last vertex gives order 0: empty features and weight Z0
    arma::vec taus1 = {0.5 * beta};
    auto zero = PackNewFeatures(taus1, 1, "remove");
    CHECK(std::get<0>(zero) == 0);
    CHECK(std::get<1>(zero).n_elem == 0);
    CHECK(std::get<5>(zero).n_elem == 0);
    CHECK(std::get<6>(zero) == Z0);

    // "stay" at order 0 also returns Z0
    arma::vec taus0;
    auto zero_stay = PackNewFeatures(taus0, 0, "stay");
    CHECK(std::get<0>(zero_stay) == 0);
    CHECK(std::get<6>(zero_stay) == Z0);
}

int main()
{
    struct { const char* name; void (*fn)(); } tests[] = {
        {"run_simulation_output", test_run_simulation_output},
        {"deterministic_with_fixed_seed", test_deterministic_with_fixed_seed},
        {"mc_move_decision_accepts_ratio_ge_one", test_mc_move_decision_accepts_ratio_ge_one},
        {"pack_new_features", test_pack_new_features},
    };
    for (auto& t : tests) {
        int before = g_failures;
        t.fn();
        std::cout << (g_failures == before ? "[PASS] " : "[FAIL] ") << t.name << std::endl;
    }
    std::cout << g_checks - g_failures << "/" << g_checks << " checks passed" << std::endl;
    return g_failures == 0 ? 0 : 1;
}
