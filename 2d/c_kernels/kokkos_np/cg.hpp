#pragma once
#include "kokkos_shared.hpp"
#include "../../shared.h"

// Initialises U
template <class Device>
struct CGInitU
{
    typedef Device device_type;

    CGInitU(
            const int x, const int y, const int coefficient, KView p, 
            KView r, KView u, KView w, KView density, KView energy) 
        : x(x), y(y), coefficient(coefficient), p(p), w(w), r(r), 
        u(u), density(density), energy(energy) { }

    KOKKOS_INLINE_FUNCTION
    void operator()(const int index) const 
    {
        const int kk = index % x; 
        const int jj = index / x; 
        p(index) = 0.0;
        r(index) = 0.0;
        u(index) = energy(index)*density(index);
        if(jj > 0 && jj < y-1 && kk > 0 & kk < x-1) {
          w(index) = (coefficient == CONDUCTIVITY) 
            ? density(index) : 1.0/density(index);
        }
    }

    const int x;
    const int y;
    const int coefficient;
    KView r;
    KView u;
    KView p;
    KView w;
    KView density;
    KView energy;
};

// Initialises Kx and Ky
template <class Device>
struct CGInitK
{
    typedef Device device_type;

    CGInitK(const int x, const int y, const int halo_depth, KView w, 
            KView kx, KView ky, const double rx, const double ry) 
        : x(x), y(y), halo_depth(halo_depth), w(w), kx(kx), 
        ky(ky), rx(rx), ry(ry) {}

    KOKKOS_INLINE_FUNCTION
    void operator()(const int index) const 
    {
        const int kk = index % x; 
        const int jj = index / x; 

        if(jj >= halo_depth && jj < y-1 && kk >= halo_depth && kk < x-1)
        {
            kx(index) = rx*(w(index-1)+w(index)) /
                (2.0*w(index-1)*w(index));
            ky(index) = ry*(w(index-x)+w(index)) /
                (2.0*w(index-x)*w(index));
        }
    }

    const int x;
    const int y;
    const int halo_depth;
    const double rx;
    const double ry;
    KView w;
    KView kx;
    KView ky;
};

// Calculates RRO
template <class Device>
struct CGInitOthers
{
    typedef double value_type;
    typedef Device device_type;

    CGInitOthers(const int x, const int y, const int halo_depth, KView kx, 
            KView ky, KView p, KView r, KView u, KView w) 
        : x(x), y(y), halo_depth(halo_depth), kx(kx), ky(ky), 
        p(p), r(r), u(u), w(w) {}

    KOKKOS_INLINE_FUNCTION
        void operator()(const int index, value_type& rro) const
        {
            const int kk = index % x; 
            const int jj = index / x; 

            if(kk >= halo_depth && kk < x - halo_depth &&
                    jj >= halo_depth && jj < y - halo_depth)
            {
                const double smvp = SMVP(u);
                w(index) = smvp;
                r(index) = u(index)-w(index);
                p(index) = r(index);
                rro += r(index)*p(index);
            }
        }

    const int x;
    const int y;
    const int halo_depth;
    KView w;
    KView p;
    KView r;
    KView u;
    KView kx;
    KView ky;
};

// Calculates W
template <class Device>
struct CGCalcW
{
    typedef double value_type;
    typedef Device device_type;
    typedef Kokkos::TeamPolicy<Device> team_policy;
    typedef typename team_policy::member_type team_member;

    CGCalcW(const int x, const int y, const int halo_depth, KView w, 
            KView p, KView kx, KView ky) 
        : x(x), y(y), halo_depth(halo_depth), w(w), p(p), 
        kx(kx), ky(ky) {}

    KOKKOS_INLINE_FUNCTION
    void operator()(const team_member& team, value_type& pw) const
    {
        double pw_team = 0.0;
        const int team_offset = (team.league_rank() + halo_depth)*x;

        Kokkos::parallel_reduce(
                Kokkos::TeamThreadRange(team, halo_depth, x-halo_depth),
                [&] (const int &j, double& pw_thread)
        {
            const int index = team_offset + j;
            const double smvp = SMVP(p);
            w(index) = smvp;
            pw_thread += smvp*p(index);
        }, pw_team);

        Kokkos::single(Kokkos::PerTeam(team), [&] ()
        {
            pw += pw_team;
        });
    }

    const int x;
    const int y;
    const int halo_depth;
    KView w;
    KView p;
    KView kx;
    KView ky;
};

// Calculates UR
template <class Device>
struct CGCalcUR
{
    typedef double value_type;
    typedef Device device_type;
    typedef Kokkos::TeamPolicy<Device> team_policy;
    typedef typename team_policy::member_type team_member;

    CGCalcUR(const int x, const int y, const int halo_depth, KView u, 
            KView r, KView p, KView w, const double alpha) 
        : x(x), y(y), halo_depth(halo_depth), u(u), r(r), 
        p(p), w(w), alpha(alpha) {}

    KOKKOS_INLINE_FUNCTION
    void operator()(const team_member& team, value_type& rrn) const
    {
        double rrn_team = 0.0;
        const int team_offset = (team.league_rank() + halo_depth)*x;

        Kokkos::parallel_reduce(
                Kokkos::TeamThreadRange(team, halo_depth, x-halo_depth),
                [&] (const int &j, double& rrn_thread)
        {
            const int index = team_offset + j;
            u(index) += alpha*p(index);
            r(index) -= alpha*w(index);
            rrn_thread += r(index)*r(index);
        }, rrn_team);

        Kokkos::single(Kokkos::PerTeam(team), [&] ()
        {
            rrn += rrn_team;
        });
    }

    const int x;
    const int y;
    const int halo_depth;
    const double alpha;
    KView u;
    KView r;
    KView p;
    KView w;
};

// Calculates P
template <class Device>
struct CGCalcP
{
    typedef Device device_type;
    typedef Kokkos::TeamPolicy<Device> team_policy;
    typedef typename team_policy::member_type team_member;

    CGCalcP(const int x, const int y, const int halo_depth, const double beta, 
            KView p, KView r) 
        : x(x), y(y), halo_depth(halo_depth), p(p), r(r), beta(beta) {}

    KOKKOS_INLINE_FUNCTION
    void operator()(const team_member& team) const
    {
        const int team_offset = (team.league_rank() + halo_depth)*x;

        Kokkos::parallel_for(
            Kokkos::TeamThreadRange(team, halo_depth, x-halo_depth),
            [&] (const int &j)
        {
            const int index = team_offset + j;
            p(index) = beta*p(index) + r(index);
        });
    }

    const int x;
    const int y;
    const int halo_depth;
    const double beta;
    KView p;
    KView r;
};

