# Prebuilt-binary spec for cortex.
#
# Unlike cortex.spec (which builds from source via cargo), this spec
# wraps a pre-built `cortex` binary produced by an upstream CI job and
# packages it for rpm.lair.cafe. The %build phase is a no-op.
#
# Required defines at rpmbuild time:
#   cortex_version    e.g. "0.1.16"
#   cortex_prerelease e.g. "0.1.20260518140530.gitabcdef0"
#                            ^^^^^^^^^^^^^^^^^^ ^^^^^^^^
#                            commit time (sec)  commit sha
#                           (used as Release; the timestamp prefix
#                            keeps same-day builds strictly ordered.)

%global _build_id_links none
%global debug_package %{nil}
%global __strip /usr/bin/true

%{!?cortex_version: %global cortex_version 0.0.0}
%if 0%{?cortex_prerelease:1}
%global cortex_release %{cortex_prerelease}
%else
%global cortex_release 1
%endif

Name:           cortex
Version:        %{cortex_version}
Release:        %{cortex_release}%{?dist}
Summary:        Inference gateway for multi-node GPU clusters (prebuilt)

License:        GPL-3.0-or-later
URL:            https://git.lair.cafe/helexa/cortex

Source0:        cortex
Source1:        cortex.service
Source2:        cortex-sysusers.conf
Source3:        cortex-firewalld.xml
Source4:        cortex.example.toml
Source5:        models.example.toml
Source6:        LICENSE

ExclusiveArch:  x86_64

Requires(pre):  shadow-utils
Requires:       systemd
Requires:       firewalld-filesystem

Provides:       user(cortex)

%description
Cortex is a Rust reverse-proxy that sits in front of multiple neuron
inference daemons and presents a unified OpenAI and Anthropic
compatible API surface.

This package wraps a binary built upstream in CI; the source-build
spec (cortex.spec) remains available for stable releases.

%prep
cp %{SOURCE0} ./cortex
cp %{SOURCE1} .
cp %{SOURCE2} .
cp %{SOURCE3} .
cp %{SOURCE4} .
cp %{SOURCE5} .
cp %{SOURCE6} .

%build
# Already built in the upstream CI build job.

%install
install -Dm755 cortex %{buildroot}%{_bindir}/cortex
install -Dm644 cortex.service %{buildroot}%{_unitdir}/cortex.service
install -Dm644 cortex-sysusers.conf %{buildroot}%{_sysusersdir}/cortex.conf
install -Dm644 cortex-firewalld.xml %{buildroot}%{_prefix}/lib/firewalld/services/cortex.xml
install -dm755 %{buildroot}%{_sysconfdir}/cortex
install -Dm644 cortex.example.toml %{buildroot}%{_sysconfdir}/cortex/cortex.toml
install -Dm644 models.example.toml %{buildroot}%{_sysconfdir}/cortex/models.toml

%pre
getent group cortex >/dev/null || groupadd -r cortex
getent passwd cortex >/dev/null || \
    useradd -r -g cortex -d /var/lib/cortex -s /sbin/nologin \
        -c "Cortex inference gateway" cortex

%post
%systemd_post cortex.service

%preun
%systemd_preun cortex.service

%postun
%systemd_postun_with_restart cortex.service

%files
%license LICENSE
%{_bindir}/cortex
%{_unitdir}/cortex.service
%{_sysusersdir}/cortex.conf
%{_prefix}/lib/firewalld/services/cortex.xml
%dir %{_sysconfdir}/cortex
%config(noreplace) %{_sysconfdir}/cortex/cortex.toml
%config(noreplace) %{_sysconfdir}/cortex/models.toml

%changelog
* Mon May 18 2026 Gitea Actions <actions@git.lair.cafe> - %{cortex_version}-%{cortex_release}
- Prerelease build from upstream CI binary.
