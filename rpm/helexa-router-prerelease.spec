# Prebuilt-binary spec for helexa-router.
#
# Wraps a pre-built `helexa-router` binary produced by an upstream CI
# job and packages it for rpm.lair.cafe. The %build phase is a no-op.
# helexa-router is a pure-Rust, non-CUDA daemon: the federation data
# plane. It serves an inbound plaintext HTTP API on tcp/8088 (edge nginx
# terminates client TLS in front of it) — hence the firewalld service —
# and connects out to each configured operator cortex, polling topology
# and dispatching inference on capacity.
#
# Required defines at rpmbuild time:
#   router_version    e.g. "0.1.16"
#   router_prerelease e.g. "0.1.20260518140530.gitabcdef0"
#                            ^^^^^^^^^^^^^^^^^^ ^^^^^^^^
#                            commit time (sec)  commit sha
#                           (used as Release; the timestamp prefix
#                            keeps same-day builds strictly ordered.)

%global _build_id_links none
%global debug_package %{nil}
%global __strip /usr/bin/true

%{!?router_version: %global router_version 0.0.0}
%if 0%{?router_prerelease:1}
%global router_release %{router_prerelease}
%else
%global router_release 1
%endif

Name:           helexa-router
Version:        %{router_version}
Release:        %{router_release}%{?dist}
Summary:        Federation data plane for helexa (prebuilt)

License:        GPL-3.0-or-later
URL:            https://git.lair.cafe/helexa/helexa

Source0:        helexa-router
Source1:        helexa-router.service
Source2:        helexa-router-sysusers.conf
Source3:        helexa-router.example.toml
Source4:        LICENSE
Source5:        helexa-router-firewalld.xml

Requires:       firewalld-filesystem

ExclusiveArch:  x86_64

Requires(pre):  shadow-utils
Requires:       systemd

Provides:       user(helexa-router)

%description
helexa-router is the federation data plane: the public /v1 ingress that
polls each operator cortex's health and model topology, resolves
product-tier aliases to real model ids, and dispatches each inference
request to a capacity-bearing cortex (warm-first, region-affine, with
transport-failure failover). It forwards the client's bearer verbatim —
entitlement enforcement stays at cortex — and can pin each cortex's
outbound TLS to an enrolled trust anchor.

%prep
cp %{SOURCE0} ./helexa-router
cp %{SOURCE1} .
cp %{SOURCE2} .
cp %{SOURCE3} .
cp %{SOURCE4} .
cp %{SOURCE5} .

%build
# Already built in the upstream CI build job.

%install
install -Dm755 helexa-router %{buildroot}%{_bindir}/helexa-router
install -Dm644 helexa-router.service %{buildroot}%{_unitdir}/helexa-router.service
install -Dm644 helexa-router-sysusers.conf %{buildroot}%{_sysusersdir}/helexa-router.conf
install -Dm644 helexa-router-firewalld.xml %{buildroot}%{_prefix}/lib/firewalld/services/helexa-router.xml
install -dm755 %{buildroot}%{_sysconfdir}/helexa-router
install -Dm644 helexa-router.example.toml %{buildroot}%{_sysconfdir}/helexa-router/helexa-router.toml

%pre
getent group helexa-router >/dev/null || groupadd -r helexa-router
getent passwd helexa-router >/dev/null || \
    useradd -r -g helexa-router -d /var/lib/helexa-router -s /sbin/nologin \
        -c "helexa-router data plane" helexa-router

%post
%systemd_post helexa-router.service

%preun
%systemd_preun helexa-router.service

%postun
%systemd_postun_with_restart helexa-router.service

%files
%license LICENSE
%{_bindir}/helexa-router
%{_unitdir}/helexa-router.service
%{_sysusersdir}/helexa-router.conf
%{_prefix}/lib/firewalld/services/helexa-router.xml
%dir %{_sysconfdir}/helexa-router
%config(noreplace) %{_sysconfdir}/helexa-router/helexa-router.toml

%changelog
* Wed Jul 16 2026 Gitea Actions <actions@git.lair.cafe> - %{router_version}-%{router_release}
- Prerelease build from upstream CI binary.
