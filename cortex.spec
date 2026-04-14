Name:           cortex
Version:        0.1.0
Release:        1%{?dist}
Summary:        Inference gateway for multi-node mistral.rs clusters

License:        GPL-3.0-or-later
URL:            https://git.lair.cafe/helexa/cortex
Source0:        %{name}-%{version}.tar.gz
Source1:        %{name}-%{version}-vendor.tar.gz

ExclusiveArch:  x86_64

BuildRequires:  rust >= 1.85
BuildRequires:  cargo
BuildRequires:  gcc
BuildRequires:  systemd-rpm-macros

%description
Cortex is a Rust reverse-proxy that sits in front of multiple mistral.rs
inference nodes and presents a unified OpenAI and Anthropic compatible
API surface. It handles model routing, lifecycle management, request
translation, and metrics collection.

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

%files
%license LICENSE
%doc README.md
%{_bindir}/cortex

%changelog
* Mon Apr 14 2026 Rob Thijssen <grenade@rob.tn> - 0.1.0-1
- Initial package
