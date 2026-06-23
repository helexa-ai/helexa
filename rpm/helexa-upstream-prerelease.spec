# Prebuilt-binary spec for helexa-upstream.
#
# Wraps a pre-built `helexa-upstream` binary produced by an upstream CI
# job and packages it for rpm.lair.cafe. The %build phase is a no-op.
# helexa-upstream is a pure-Rust, non-CUDA daemon: the mesh account +
# budget authority. It serves an inbound HTTP API on tcp/8090 (/authz/v1
# for cortex, /web/v1 for the frontend) — hence the firewalld service —
# and connects out to PostgreSQL (the system of record), running schema
# migrations on startup.
#
# Required defines at rpmbuild time:
#   upstream_version    e.g. "0.1.16"
#   upstream_prerelease e.g. "0.1.20260518140530.gitabcdef0"
#                              ^^^^^^^^^^^^^^^^^^ ^^^^^^^^
#                              commit time (sec)  commit sha
#                             (used as Release; the timestamp prefix
#                              keeps same-day builds strictly ordered.)

%global _build_id_links none
%global debug_package %{nil}
%global __strip /usr/bin/true

%{!?upstream_version: %global upstream_version 0.0.0}
%if 0%{?upstream_prerelease:1}
%global upstream_release %{upstream_prerelease}
%else
%global upstream_release 1
%endif

Name:           helexa-upstream
Version:        %{upstream_version}
Release:        %{upstream_release}%{?dist}
Summary:        Mesh account + budget authority for helexa (prebuilt)

License:        GPL-3.0-or-later
URL:            https://git.lair.cafe/helexa/helexa

Source0:        helexa-upstream
Source1:        helexa-upstream.service
Source2:        helexa-upstream-sysusers.conf
Source3:        helexa-upstream.example.toml
Source4:        LICENSE
Source5:        helexa-upstream-firewalld.xml

Requires:       firewalld-filesystem

ExclusiveArch:  x86_64

Requires(pre):  shadow-utils
Requires:       systemd

Provides:       user(helexa-upstream)

%description
helexa-upstream is the mesh-level authority: it issues user accounts and
API keys, holds the allocation ledger (free grant + redeemable top-up
codes), enforces per-key budgets via a transactional reserve/settle
contract, and reconciles served usage for operator compensation. cortex
validates locally-unrecognised keys against its /authz/v1 surface
(fail-closed); the helexa.ai frontend drives account self-service via
/web/v1. PostgreSQL is the system of record; the schema is migrated on
startup.

%prep
cp %{SOURCE0} ./helexa-upstream
cp %{SOURCE1} .
cp %{SOURCE2} .
cp %{SOURCE3} .
cp %{SOURCE4} .
cp %{SOURCE5} .

%build
# Already built in the upstream CI build job.

%install
install -Dm755 helexa-upstream %{buildroot}%{_bindir}/helexa-upstream
install -Dm644 helexa-upstream.service %{buildroot}%{_unitdir}/helexa-upstream.service
install -Dm644 helexa-upstream-sysusers.conf %{buildroot}%{_sysusersdir}/helexa-upstream.conf
install -Dm644 helexa-upstream-firewalld.xml %{buildroot}%{_prefix}/lib/firewalld/services/helexa-upstream.xml
install -dm755 %{buildroot}%{_sysconfdir}/helexa-upstream
install -Dm644 helexa-upstream.example.toml %{buildroot}%{_sysconfdir}/helexa-upstream/helexa-upstream.toml

%pre
getent group helexa-upstream >/dev/null || groupadd -r helexa-upstream
getent passwd helexa-upstream >/dev/null || \
    useradd -r -g helexa-upstream -d /var/lib/helexa-upstream -s /sbin/nologin \
        -c "helexa-upstream authority" helexa-upstream

%post
%systemd_post helexa-upstream.service

%preun
%systemd_preun helexa-upstream.service

%postun
%systemd_postun_with_restart helexa-upstream.service

%files
%license LICENSE
%{_bindir}/helexa-upstream
%{_unitdir}/helexa-upstream.service
%{_sysusersdir}/helexa-upstream.conf
%{_prefix}/lib/firewalld/services/helexa-upstream.xml
%dir %{_sysconfdir}/helexa-upstream
%config(noreplace) %{_sysconfdir}/helexa-upstream/helexa-upstream.toml

%changelog
* Mon Jun 23 2026 Gitea Actions <actions@git.lair.cafe> - %{upstream_version}-%{upstream_release}
- Prerelease build from upstream CI binary.
