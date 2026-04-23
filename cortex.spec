Name:           cortex
Version:        0.1.16
Release:        1%{?dist}
Summary:        Inference gateway for multi-node GPU clusters

License:        GPL-3.0-or-later
URL:            https://git.lair.cafe/helexa/cortex
Source0:        %{name}-%{version}.tar.gz
Source1:        %{name}-%{version}-vendor.tar.gz

ExclusiveArch:  x86_64

BuildRequires:  rust >= 1.85
BuildRequires:  cargo
BuildRequires:  gcc
BuildRequires:  gcc-c++
BuildRequires:  cmake
BuildRequires:  perl-interpreter
BuildRequires:  pkgconfig(openssl)
BuildRequires:  systemd-rpm-macros

Requires(pre):  shadow-utils
Requires:       systemd
Requires:       firewalld-filesystem

# systemd-rpm-macros ships a unit dep generator that parses User=/Group=
# from our .service file and emits Requires: user(cortex)/group(cortex).
# rpm's sysusers provides-generator emits the unversioned form for groups
# but only a versioned user(cortex) = <base64> for users with GECOS/home/
# shell. Provide the unversioned user(cortex) explicitly so dnf can resolve
# the auto-generated Requires. Without this, dnf5 silently filters the
# package and reports "Nothing to do".
Provides:       user(cortex)

%description
Cortex is a Rust reverse-proxy that sits in front of multiple inference
nodes (via neuron daemons) and presents a unified OpenAI and Anthropic
compatible API surface. It handles model routing, lifecycle management,
request translation, and metrics collection.

%prep
%autosetup
tar xf %{SOURCE1}
mkdir -p .cargo
cat > .cargo/config.toml << 'EOF'
[source.crates-io]
replace-with = "vendored-sources"

[source.vendored-sources]
directory = "vendor"
EOF

%build
cargo build --release -p cortex-cli

%install
install -Dm755 target/release/cortex %{buildroot}%{_bindir}/cortex
install -Dm644 data/cortex.service %{buildroot}%{_unitdir}/cortex.service
install -Dm644 data/cortex-sysusers.conf %{buildroot}%{_sysusersdir}/cortex.conf
install -Dm644 data/cortex-firewalld.xml %{buildroot}%{_prefix}/lib/firewalld/services/cortex.xml
install -dm755 %{buildroot}%{_sysconfdir}/cortex
install -Dm644 cortex.example.toml %{buildroot}%{_sysconfdir}/cortex/cortex.toml
install -Dm644 models.example.toml %{buildroot}%{_sysconfdir}/cortex/models.toml

%pre
%sysusers_create_compat %{_builddir}/%{name}-%{version}/data/cortex-sysusers.conf

%post
%systemd_post cortex.service

%preun
%systemd_preun cortex.service

%postun
%systemd_postun_with_restart cortex.service

%files
%license LICENSE
%doc README.md
%{_bindir}/cortex
%{_unitdir}/cortex.service
%{_sysusersdir}/cortex.conf
%{_prefix}/lib/firewalld/services/cortex.xml
%dir %{_sysconfdir}/cortex
%config(noreplace) %{_sysconfdir}/cortex/cortex.toml
%config(noreplace) %{_sysconfdir}/cortex/models.toml

%changelog
* Thu Apr 16 2026 Gitea Actions <actions@git.lair.cafe> - 0.1.16-1
- chore: ignore local deploy script
- chore: move default ports out of common-collision ranges
- ci: drop actions/cache for cargo registry and target

* Thu Apr 16 2026 Gitea Actions <actions@git.lair.cafe> - 0.1.14-1
- ci: publish both packages to a single helexa/helexa COPR project
- fix(rpm): rename neuron package to helexa-neuron
- ci: commit generated %changelog entries back to main

* Wed Apr 15 2026 Rob Thijssen <grenade@rob.tn> - 0.1.0-1
- Initial package
