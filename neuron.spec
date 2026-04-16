Name:           neuron
Version:        0.1.5
Release:        1%{?dist}
Summary:        Per-node GPU discovery and harness management daemon for cortex

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

# rpm's sysusers provides-generator only emits versioned user(cortex) when
# the u-line has GECOS/home/shell fields. %attr(,,cortex) in %files emits
# an unversioned Requires: user(cortex), so we provide it explicitly.
Provides:       user(cortex)
Provides:       group(cortex)

%description
Neuron is a per-node daemon for cortex inference clusters. It discovers
local GPU hardware via nvidia-smi, manages inference harnesses (mistral.rs,
llama.cpp), and exposes an HTTP API for model lifecycle management.

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
cargo build --release -p neuron

%install
install -Dm755 target/release/neuron %{buildroot}%{_bindir}/neuron
install -Dm644 data/neuron.service %{buildroot}%{_unitdir}/neuron.service
install -Dm644 data/cortex-sysusers.conf %{buildroot}%{_sysusersdir}/neuron.conf
install -dm750 %{buildroot}%{_sysconfdir}/neuron
install -Dm640 neuron.example.toml %{buildroot}%{_sysconfdir}/neuron/neuron.toml

%pre
%sysusers_create_compat %{_builddir}/%{name}-%{version}/data/cortex-sysusers.conf

%post
%systemd_post neuron.service

%preun
%systemd_preun neuron.service

%postun
%systemd_postun_with_restart neuron.service

%files
%license LICENSE
%doc README.md
%{_bindir}/neuron
%{_unitdir}/neuron.service
%{_sysusersdir}/neuron.conf
%dir %attr(750,root,cortex) %{_sysconfdir}/neuron
%config(noreplace) %attr(640,root,cortex) %{_sysconfdir}/neuron/neuron.toml

%changelog
* Tue Apr 15 2026 Rob Thijssen <grenade@rob.tn> - 0.1.0-1
- Initial package
