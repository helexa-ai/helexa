# Prebuilt-binary spec for helexa-angels.
#
# Wraps a pre-built `helexa-angels` binary produced by an upstream CI job
# and packages it for rpm.lair.cafe. The %build phase is a no-op.
# helexa-angels is a pure-Rust, non-CUDA daemon: the confidential investor
# portal behind angels.helexa.ai. It serves server-rendered HTML on
# tcp/8092 to the edge proxies only (hence the firewalld service) and
# connects out to the PostgreSQL helexa-upstream owns — credential auth is
# shared, so an investor has one helexa account for both properties. It
# runs its own migrations, confined to the `angels` schema, on startup.
#
# Note it packages NO content. Round documents are deployed separately
# into /var/lib/helexa-angels/content because they are confidential and
# must never enter the open-source repository this RPM is built from.
#
# Required defines at rpmbuild time:
#   upstream_version    e.g. "0.1.16"
#   upstream_prerelease e.g. "0.1.20260518140530.gitabcdef0"

%global _build_id_links none
%global debug_package %{nil}
%global __strip /usr/bin/true

%{!?upstream_version: %global upstream_version 0.0.0}
%if 0%{?upstream_prerelease:1}
%global upstream_release %{upstream_prerelease}
%else
%global upstream_release 1
%endif

Name:           helexa-angels
Version:        %{upstream_version}
Release:        %{upstream_release}%{?dist}
Summary:        Confidential investor portal for helexa (prebuilt)

License:        GPL-3.0-or-later
URL:            https://git.lair.cafe/helexa/helexa

Source0:        helexa-angels
Source1:        helexa-angels.service
Source2:        helexa-angels-sysusers.conf
Source3:        helexa-angels.example.toml
Source4:        LICENSE
Source5:        helexa-angels-firewalld.xml

Requires:       firewalld-filesystem

ExclusiveArch:  x86_64

Requires(pre):  shadow-utils
Requires:       systemd

Provides:       user(helexa-angels)

%description
helexa-angels serves angels.helexa.ai: confidential material prepared for
named investors, behind invitation codes and per-round grants. Pages are
assembled server-side so that nothing an unauthenticated request can reach
contains round content, every document view is attributed to a named
account in an access log, and each page carries a per-viewer watermark.
Credentials are shared with helexa.ai via the same PostgreSQL `users`
table; sessions deliberately are not.

%prep
cp %{SOURCE0} ./helexa-angels
cp %{SOURCE1} .
cp %{SOURCE2} .
cp %{SOURCE3} .
cp %{SOURCE4} .
cp %{SOURCE5} .

%build
# Already built in the upstream CI build job.

%install
install -Dm755 helexa-angels %{buildroot}%{_bindir}/helexa-angels
install -Dm644 helexa-angels.service %{buildroot}%{_unitdir}/helexa-angels.service
install -Dm644 helexa-angels-sysusers.conf %{buildroot}%{_sysusersdir}/helexa-angels.conf
install -Dm644 helexa-angels-firewalld.xml %{buildroot}%{_prefix}/lib/firewalld/services/helexa-angels.xml
install -dm755 %{buildroot}%{_sysconfdir}/helexa-angels
install -Dm644 helexa-angels.example.toml %{buildroot}%{_sysconfdir}/helexa-angels/helexa-angels.toml
# Content directory. Mode 0750 and owned by the service user: the files
# that land here are the confidential material itself.
install -dm750 %{buildroot}%{_sharedstatedir}/helexa-angels
install -dm750 %{buildroot}%{_sharedstatedir}/helexa-angels/content

%pre
getent group helexa-angels >/dev/null || groupadd -r helexa-angels
getent passwd helexa-angels >/dev/null || \
    useradd -r -g helexa-angels -d /var/lib/helexa-angels -s /sbin/nologin \
        -c "helexa-angels investor portal" helexa-angels

%post
%systemd_post helexa-angels.service
# The config carries a database URL. %config(noreplace) keeps an existing
# file as-is on upgrade — including a too-permissive mode from an older
# package — so converge it here.
if [ -f %{_sysconfdir}/helexa-angels/helexa-angels.toml ]; then
    chgrp helexa-angels %{_sysconfdir}/helexa-angels/helexa-angels.toml >/dev/null 2>&1 || :
    chmod 0640 %{_sysconfdir}/helexa-angels/helexa-angels.toml >/dev/null 2>&1 || :
fi
# Same for the content tree: an upgrade must not loosen it.
if [ -d %{_sharedstatedir}/helexa-angels/content ]; then
    chown -R helexa-angels:helexa-angels %{_sharedstatedir}/helexa-angels >/dev/null 2>&1 || :
    chmod 0750 %{_sharedstatedir}/helexa-angels %{_sharedstatedir}/helexa-angels/content >/dev/null 2>&1 || :
fi

%preun
%systemd_preun helexa-angels.service

%postun
%systemd_postun_with_restart helexa-angels.service

%files
%license LICENSE
%{_bindir}/helexa-angels
%{_unitdir}/helexa-angels.service
%{_sysusersdir}/helexa-angels.conf
%{_prefix}/lib/firewalld/services/helexa-angels.xml
%dir %{_sysconfdir}/helexa-angels
%config(noreplace) %attr(0640,root,helexa-angels) %{_sysconfdir}/helexa-angels/helexa-angels.toml
%dir %attr(0750,helexa-angels,helexa-angels) %{_sharedstatedir}/helexa-angels
%dir %attr(0750,helexa-angels,helexa-angels) %{_sharedstatedir}/helexa-angels/content

%changelog
* Sun Aug 02 2026 Gitea Actions <actions@git.lair.cafe> - %{upstream_version}-%{upstream_release}
- Prerelease build from upstream CI binary.
