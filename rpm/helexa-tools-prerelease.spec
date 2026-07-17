# Prebuilt-binary spec for helexa-tools.
#
# Wraps a pre-built `helexa-tools` binary produced by an upstream CI
# job and packages it for rpm.lair.cafe. The %build phase is a no-op.
# helexa-tools is a pure-Rust, non-CUDA daemon: the tool-execution
# service backing the chat app's grounding tools (#177). It serves an
# inbound plaintext HTTP API on tcp/8889 (edge nginx fronts it as the
# rate-limited /tools/fetch) — hence the firewalld service — and
# connects out to public web pages, extracting readable article text
# behind a strict SSRF policy (public hostnames + globally routable
# addresses only, per-redirect-hop revalidation).
#
# Required defines at rpmbuild time:
#   tools_version    e.g. "0.1.16"
#   tools_prerelease e.g. "0.1.20260518140530.gitabcdef0"

%global _build_id_links none
%global debug_package %{nil}
%global __strip /usr/bin/true

%{!?tools_version: %global tools_version 0.0.0}
%if 0%{?tools_prerelease:1}
%global tools_release %{tools_prerelease}
%else
%global tools_release 1
%endif

Name:           helexa-tools
Version:        %{tools_version}
Release:        %{tools_release}%{?dist}
Summary:        Tool-execution service for helexa grounding (prebuilt)

License:        GPL-3.0-or-later
URL:            https://git.lair.cafe/helexa/helexa

Source0:        helexa-tools
Source1:        helexa-tools.service
Source2:        helexa-tools-sysusers.conf
Source3:        helexa-tools.example.toml
Source4:        LICENSE
Source5:        helexa-tools-firewalld.xml

Requires:       firewalld-filesystem

ExclusiveArch:  x86_64

Requires(pre):  shadow-utils
Requires:       systemd

Provides:       user(helexa-tools)

%description
helexa-tools executes grounding tools for the helexa chat app: an
SSRF-guarded page fetcher with readability extraction, so models can
read a web page's article text instead of guessing from search
snippets. Only http/https to public, globally-routable hosts is
fetchable; every redirect hop is re-validated.

%prep
cp %{SOURCE0} ./helexa-tools
cp %{SOURCE1} .
cp %{SOURCE2} .
cp %{SOURCE3} .
cp %{SOURCE4} .
cp %{SOURCE5} .

%build
# Already built in the upstream CI build job.

%install
install -Dm755 helexa-tools %{buildroot}%{_bindir}/helexa-tools
install -Dm644 helexa-tools.service %{buildroot}%{_unitdir}/helexa-tools.service
install -Dm644 helexa-tools-sysusers.conf %{buildroot}%{_sysusersdir}/helexa-tools.conf
install -Dm644 helexa-tools-firewalld.xml %{buildroot}%{_prefix}/lib/firewalld/services/helexa-tools.xml
install -dm755 %{buildroot}%{_sysconfdir}/helexa-tools
install -Dm644 helexa-tools.example.toml %{buildroot}%{_sysconfdir}/helexa-tools/helexa-tools.toml

%pre
getent group helexa-tools >/dev/null || groupadd -r helexa-tools
getent passwd helexa-tools >/dev/null || \
    useradd -r -g helexa-tools -d /var/lib/helexa-tools -s /sbin/nologin \
        -c "helexa-tools grounding fetcher" helexa-tools

%post
%systemd_post helexa-tools.service

%preun
%systemd_preun helexa-tools.service

%postun
%systemd_postun_with_restart helexa-tools.service

%files
%license LICENSE
%{_bindir}/helexa-tools
%{_unitdir}/helexa-tools.service
%{_sysusersdir}/helexa-tools.conf
%{_prefix}/lib/firewalld/services/helexa-tools.xml
%dir %{_sysconfdir}/helexa-tools
%config(noreplace) %{_sysconfdir}/helexa-tools/helexa-tools.toml

%changelog
* Thu Jul 17 2026 Gitea Actions <actions@git.lair.cafe> - %{tools_version}-%{tools_release}
- Prerelease build from upstream CI binary.
