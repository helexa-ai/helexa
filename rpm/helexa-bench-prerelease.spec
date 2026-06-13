# Prebuilt-binary spec for helexa-bench.
#
# Wraps a pre-built `helexa-bench` binary produced by an upstream CI job
# and packages it for rpm.lair.cafe. The %build phase is a no-op.
# helexa-bench is a pure-Rust, non-CUDA, outbound-only daemon (no
# listener), so there is no firewalld service to install.
#
# Required defines at rpmbuild time:
#   bench_version    e.g. "0.1.16"
#   bench_prerelease e.g. "0.1.20260518140530.gitabcdef0"
#                            ^^^^^^^^^^^^^^^^^^ ^^^^^^^^
#                            commit time (sec)  commit sha
#                           (used as Release; the timestamp prefix
#                            keeps same-day builds strictly ordered.)

%global _build_id_links none
%global debug_package %{nil}
%global __strip /usr/bin/true

%{!?bench_version: %global bench_version 0.0.0}
%if 0%{?bench_prerelease:1}
%global bench_release %{bench_prerelease}
%else
%global bench_release 1
%endif

Name:           helexa-bench
Version:        %{bench_version}
Release:        %{bench_release}%{?dist}
Summary:        Continuous version-aware benchmark harness for the neuron fleet (prebuilt)

License:        GPL-3.0-or-later
URL:            https://git.lair.cafe/helexa/helexa

Source0:        helexa-bench
Source1:        helexa-bench.service
Source2:        helexa-bench-sysusers.conf
Source3:        helexa-bench.example.toml
Source4:        LICENSE

ExclusiveArch:  x86_64

Requires(pre):  shadow-utils
Requires:       systemd

Provides:       user(helexa-bench)

%description
helexa-bench hits each neuron on the fleet directly, exercises an
extensible benchmark suite against every warm model, and records each
run with full build/version provenance into a SQLite store. It runs
continuously under systemd and is version-aware: a given neuron build is
benchmarked only until it has the configured number of samples, then
skipped until a new build ships. Replaces manual bench.py runs.

%prep
cp %{SOURCE0} ./helexa-bench
cp %{SOURCE1} .
cp %{SOURCE2} .
cp %{SOURCE3} .
cp %{SOURCE4} .

%build
# Already built in the upstream CI build job.

%install
install -Dm755 helexa-bench %{buildroot}%{_bindir}/helexa-bench
install -Dm644 helexa-bench.service %{buildroot}%{_unitdir}/helexa-bench.service
install -Dm644 helexa-bench-sysusers.conf %{buildroot}%{_sysusersdir}/helexa-bench.conf
install -dm755 %{buildroot}%{_sysconfdir}/helexa-bench
install -Dm644 helexa-bench.example.toml %{buildroot}%{_sysconfdir}/helexa-bench/helexa-bench.toml

%pre
getent group helexa-bench >/dev/null || groupadd -r helexa-bench
getent passwd helexa-bench >/dev/null || \
    useradd -r -g helexa-bench -d /var/lib/helexa-bench -s /sbin/nologin \
        -c "helexa-bench harness" helexa-bench

%post
%systemd_post helexa-bench.service

%preun
%systemd_preun helexa-bench.service

%postun
%systemd_postun_with_restart helexa-bench.service

%files
%license LICENSE
%{_bindir}/helexa-bench
%{_unitdir}/helexa-bench.service
%{_sysusersdir}/helexa-bench.conf
%dir %{_sysconfdir}/helexa-bench
%config(noreplace) %{_sysconfdir}/helexa-bench/helexa-bench.toml

%changelog
* Sat Jun 13 2026 Gitea Actions <actions@git.lair.cafe> - %{bench_version}-%{bench_release}
- Prerelease build from upstream CI binary.
