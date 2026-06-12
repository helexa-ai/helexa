# Prebuilt-binary spec for helexa-neuron flavoured by CUDA compute capability.
#
# Unlike helexa-neuron.spec (which builds from source via cargo), this
# spec wraps a pre-built `neuron-{flavour}` binary produced by an
# upstream CI job and packages it for rpm.lair.cafe. The %build phase
# is a no-op.
#
# Required defines at rpmbuild time:
#   neuron_version    e.g. "0.1.16"
#   neuron_flavour    e.g. "ada", "blackwell"  — matches the CI build
#                     matrix's compute_cap label.
#   neuron_prerelease e.g. "0.1.20260518140530.gitabcdef0"
#                            ^^^^^^^^^^^^^^^^^^ ^^^^^^^^
#                            commit time (sec)  commit sha
#                           (used as Release; the timestamp prefix
#                            keeps same-day builds strictly ordered.)
#
# One flavour can be installed at a time on a given host; flavour
# packages Conflict with each other.

%global _build_id_links none
%global debug_package %{nil}
%global __strip /usr/bin/true

%{!?neuron_version: %global neuron_version 0.0.0}
%{!?neuron_flavour: %global neuron_flavour blackwell}
%if 0%{?neuron_prerelease:1}
%global neuron_release %{neuron_prerelease}
%else
%global neuron_release 1
%endif

Name:           helexa-neuron-%{neuron_flavour}
Version:        %{neuron_version}
Release:        %{neuron_release}%{?dist}
Summary:        Per-node GPU inference daemon (candle, %{neuron_flavour} flavour)

License:        GPL-3.0-or-later
URL:            https://git.lair.cafe/helexa/helexa

Source0:        neuron-%{neuron_flavour}
Source1:        neuron.service
Source2:        neuron-sysusers.conf
Source3:        neuron-firewalld.xml
Source4:        neuron.example.toml
Source5:        LICENSE

ExclusiveArch:  x86_64

# Binary links against the CUDA runtime, cuDNN, NCCL, etc. Suppress
# auto-detected exact soname deps — users may have CUDA from various
# sources (rpmfusion, nvidia-direct) at different compatible versions;
# a runtime dlopen failure surfaces a clearer error than rpm dep
# resolution would.
%global __requires_exclude ^lib(cuda|cudart|cudnn|cublas|cublasLt|curand|nvrtc|nccl)

Requires(pre):  shadow-utils
Requires:       systemd
Requires:       firewalld-filesystem

Provides:       helexa-neuron = %{neuron_version}-%{neuron_release}
Provides:       user(neuron)

# Mutual exclusion across flavours and the source-build variant.
Conflicts:      helexa-neuron
Conflicts:      helexa-neuron-ada
Conflicts:      helexa-neuron-ampere
Conflicts:      helexa-neuron-blackwell
# (The Conflicts: with self is filtered by rpm at install time.)

%description
Neuron is the per-node daemon for cortex inference clusters. It
discovers local GPU hardware via nvidia-smi, runs in-process
inference via huggingface/candle, and exposes an HTTP API for model
lifecycle management (load, unload, list, inference endpoint).

This is the %{neuron_flavour} flavour, built for that CUDA compute
capability. Install the flavour matching the GPUs on this host.

%prep
cp %{SOURCE0} ./neuron
cp %{SOURCE1} .
cp %{SOURCE2} .
cp %{SOURCE3} .
cp %{SOURCE4} .
cp %{SOURCE5} .

%build
# Already built in the upstream CI build job (with --features cuda).

%install
install -Dm755 neuron %{buildroot}%{_bindir}/neuron
install -Dm644 neuron.service %{buildroot}%{_unitdir}/neuron.service
install -Dm644 neuron-sysusers.conf %{buildroot}%{_sysusersdir}/neuron.conf
install -Dm644 neuron-firewalld.xml %{buildroot}%{_prefix}/lib/firewalld/services/helexa-neuron.xml
install -dm755 %{buildroot}%{_sysconfdir}/neuron
install -Dm644 neuron.example.toml %{buildroot}%{_sysconfdir}/neuron/neuron.toml

%pre
getent group neuron >/dev/null || groupadd -r neuron
getent passwd neuron >/dev/null || \
    useradd -r -g neuron -d /var/lib/neuron -s /sbin/nologin \
        -G video,render \
        -c "Neuron GPU node daemon" neuron

%post
%systemd_post neuron.service

%preun
%systemd_preun neuron.service

%postun
%systemd_postun_with_restart neuron.service

%files
%license LICENSE
%{_bindir}/neuron
%{_unitdir}/neuron.service
%{_sysusersdir}/neuron.conf
%{_prefix}/lib/firewalld/services/helexa-neuron.xml
%dir %{_sysconfdir}/neuron
%config(noreplace) %{_sysconfdir}/neuron/neuron.toml

%changelog
* Mon May 18 2026 Gitea Actions <actions@git.lair.cafe> - %{neuron_version}-%{neuron_release}
- Prerelease build from upstream CI binary (%{neuron_flavour} flavour).
