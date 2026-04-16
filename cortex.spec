Name:           cortex
Version:        0.1.2
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
install -dm750 %{buildroot}%{_sysconfdir}/cortex
install -Dm640 cortex.example.toml %{buildroot}%{_sysconfdir}/cortex/cortex.toml
install -Dm640 models.example.toml %{buildroot}%{_sysconfdir}/cortex/models.toml

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
%dir %attr(750,root,cortex) %{_sysconfdir}/cortex
%config(noreplace) %attr(640,root,cortex) %{_sysconfdir}/cortex/cortex.toml
%config(noreplace) %attr(640,root,cortex) %{_sysconfdir}/cortex/models.toml

%changelog
* Tue Apr 15 2026 Rob Thijssen <grenade@rob.tn> - 0.1.0-1
- Initial package
