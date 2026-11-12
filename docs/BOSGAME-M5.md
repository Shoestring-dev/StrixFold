# Setting up a Bosgame M5 for StrixFold

The Bosgame M5 has the same chip as the boxes StrixFold was measured on, AMD's Ryzen AI Halo developer boxes: an AMD
Ryzen AI Max+ 395 with the Radeon 8060S (gfx1151) and 128 GB of memory. It ships with Windows, though, and its Secure
Boot firmware doesn't trust the kernel the tested boxes run. These steps take a new M5 to the same Linux, kernel and
GPU stack as the tested boxes (see [Tested on](../README.md#tested-on)), with Secure Boot left on.

We set up our own M5s this way in November 2026. We haven't measured StrixFold's speed on a pair of M5s yet; if you do,
please post your numbers (see [COMMUNITY-RESULTS.md](COMMUNITY-RESULTS.md)).

**You need:** a USB stick of 1 GB or more with the Debian 13 netinst image for amd64
([cdimage.debian.org](https://cdimage.debian.org/debian-cd/current/amd64/iso-cd/)) written to it, a monitor and a USB
keyboard, an Ethernet cable, and about 30 minutes per box. The monitor and keyboard are needed for steps 1 and 2 and
for the restart in step 6; the rest can be done over SSH.

## 1. BIOS

Turn the box on and tap **Del** to open the setup (AMI Aptio). The labels below are the usual ones; yours may differ
slightly.

| Setting | Where to look | Set to |
|---|---|---|
| UMA Frame Buffer Size (iGPU memory) | Advanced | the smallest value (512M or 1G). Linux gives the GPU its memory instead (step 5) |
| Restore on AC power loss / AC Loss Control | Advanced | Power On (optional; the box starts again after a power cut) |
| Fast Boot | Boot | Disabled, so the box sees the USB stick at start-up |
| Secure Boot | Security | leave it on |

Then go to **Save & Exit**, choose **Save Changes**, and under **Boot Override** pick the entry starting with
"UEFI:" and naming your USB stick.

## 2. Install Debian 13

The box starts the installer from the stick. Choose **Install** (the text installer).

- **Root password:** leave it empty. Your user is then allowed to use `sudo`.
- **Partitioning:** Guided, use entire disk. Pick the internal NVMe disk, not the USB stick. All files in one
  partition. This removes Windows.
- **Software selection:** untick "Debian desktop environment" and "GNOME", and tick **SSH server**.

When it finishes, remove the stick and let the box restart.

## 3. AMD's package repository

The Ryzen AI Halo boxes run Debian 13 with an extra repository, built by Collabora. It carries the kernel, firmware
and ROCm that StrixFold was tested with. The repository is AMD and Collabora's, made for their own boxes; it isn't a
supported path for the M5, though it worked on ours.

Its signing key comes as a small package; check its hash before installing it, then the key itself:

```bash
cd /tmp
wget https://debs.ryai.dev/pool/main/r/ryai-archive-keyring/ryai-archive-keyring_0.2~0.1+1+ge0afb86bvdebian13.1_all.deb
echo "0fcf1e841ae13f85fc5d1e0b836db1260e5fa07ebe1c90ff1836139bb7f49e49  ryai-archive-keyring_0.2~0.1+1+ge0afb86bvdebian13.1_all.deb" | sha256sum -c
sudo apt install gpg ./ryai-archive-keyring_0.2~0.1+1+ge0afb86bvdebian13.1_all.deb
gpg --show-keys /usr/share/keyrings/ryai-archive-keyring.gpg
```

`gpg` must show "Nimbus archive key" with the fingerprint `B83B6BF3716E12EE71BEA0EEDD4539D281C3A637`. If it doesn't,
stop. Otherwise add the repository:

```bash
sudo tee /etc/apt/sources.list.d/ryai.sources >/dev/null <<'EOF'
Types: deb
URIs: https://debs.ryai.dev/
Suites: rex
Components: main non-free
Signed-By: /usr/share/keyrings/ryai-archive-keyring.gpg
EOF
sudo tee /etc/apt/sources.list.d/ryai-security.sources >/dev/null <<'EOF'
Types: deb
URIs: https://security.ryai.dev/
Suites: rex-security
Components: main
Signed-By: /usr/share/keyrings/ryai-archive-keyring.gpg
EOF
sudo apt update
```

Install only the packages named below: the repository also rebuilds desktop packages you don't need.

## 4. Kernel 7.2.7 and firmware

```bash
sudo apt install linux-image-7.2.7+rex-amd64=7.2.7-1+rex1bvdebian13.1 \
  firmware-amd-graphics=20260519-1~bpo13+1+rex1bvdebian13.1 \
  amd64-microcode=3.20251202.1~bpo13+0+nimbus1bvdebian13.1
sudo apt-mark hold linux-image-7.2.7+rex-amd64 firmware-amd-graphics amd64-microcode
```

The hold keeps the box on the tested stack, but it also stops security updates to these three packages. To take one,
run `sudo apt-mark unhold linux-image-7.2.7+rex-amd64 firmware-amd-graphics amd64-microcode`, upgrade, and hold them
again.

Don't restart yet: with Secure Boot on, the M5 won't start this kernel until step 6.

## 5. Give the GPU 124 GiB

Set the kernel options in `/etc/default/grub` and rebuild the boot menu. The `sed` replaces the whole
`GRUB_CMDLINE_LINUX_DEFAULT` line, which holds only `quiet` after a fresh install:

```bash
sudo sed -i 's/^GRUB_CMDLINE_LINUX_DEFAULT=.*/GRUB_CMDLINE_LINUX_DEFAULT="quiet amdgpu.gttsize=126976 ttm.pages_limit=32505856"/' /etc/default/grub
grep ^GRUB_CMDLINE_LINUX_DEFAULT /etc/default/grub
sudo update-grub
sudo grep -c gttsize /boot/grub/grub.cfg    # a number above 0
```

Also stop the box from going to sleep:

```bash
sudo systemctl mask sleep.target suspend.target hibernate.target hybrid-sleep.target
```

## 6. Secure Boot: trust the kernel's key

The 7.2.7 kernel is signed with "Collabora Rex secure boot - CA 2026". The Ryzen AI Halo's firmware has that
certificate built in; the M5's doesn't, so it refuses the kernel. Adding the certificate as a Machine Owner Key keeps
Secure Boot on and trusts only this one extra signer. The certificate is taken from the signature on the kernel that
step 4 installed (`linux-image-7.2.7+rex-amd64` depends on `linux-binary-7.2.7+rex-amd64`, which holds the kernel
file), then checked against its known fingerprint:

```bash
sudo apt install sbsigntool mokutil openssl
mkdir -p /tmp/rexca && cd /tmp/rexca
sbattach --detach sig.p7 /usr/lib/modules/7.2.7+rex-amd64/vmlinuz
openssl pkcs7 -inform DER -in sig.p7 -print_certs -out certs.pem
csplit -s -z -f cert- certs.pem '/-----BEGIN CERTIFICATE-----/' '{*}'
for c in cert-*; do openssl x509 -in "$c" -noout -subject 2>/dev/null | grep -q "CA 2026" \
  && openssl x509 -in "$c" -outform DER -out rex-ca.der; done
openssl x509 -inform DER -in rex-ca.der -noout -subject -fingerprint -sha1
```

It must print `subject=CN=Collabora Rex secure boot - CA 2026` and
`sha1 Fingerprint=BC:B9:A9:01:15:AA:84:DE:F1:51:AB:0C:1A:97:84:DC:AA:11:DF:0D`, the same certificate the Ryzen AI
Halo's firmware holds. If it doesn't, stop. Otherwise:

```bash
sudo mokutil --import rex-ca.der   # asks you to make up a one-time password
sudo reboot
```

At the restart a blue **MOK management** screen appears for a few seconds. Press a key, then choose **Enroll MOK →
Continue → Yes**, type the one-time password and choose **Reboot**. If you miss the screen, run `mokutil --import`
again and restart.

The box now starts the new kernel. Check it:

```bash
uname -r                     # 7.2.7+rex-amd64
mokutil --sb-state           # SecureBoot enabled
cat /sys/class/drm/card*/device/mem_info_gtt_total   # 133143986176 (124 GiB)
```

(Turning Secure Boot off in the BIOS would also work, but you lose its protection against a tampered kernel.)

## 7. ROCm 10.0

```bash
sudo apt install amdrocm-core-sdk10.0-gfx1151=10.0.0-4 amdrocm-core10.0-gfx1151=10.0.0-4 \
  therock-gfx1151=10.0+rex1bvdebian13.1
sudo apt-mark hold $(dpkg-query -W -f '${Package}\n' 'amdrocm*') therock-gfx1151
sudo ldconfig
sudo usermod -aG render,video "$USER"
```

Log out and back in (or open a new SSH session) so the new groups apply. Until then the GPU device `/dev/kfd` is
denied and `rocminfo` fails. Then check:

```bash
/opt/rocm/core-10.0/bin/rocminfo | grep -m1 gfx1151
/opt/rocm/core-10.0/bin/hipconfig --version   # 7.15...
ldconfig -p | grep -c libamdhip64             # 2
```

`therock-gfx1151` is small but needed: it registers ROCm's libraries with the system. Without it `rocminfo` still
works, but StrixFold's programs stop with `DriverUnavailable` because they can't load the GPU runtime.

On one M5 we built StrixFold's GPU tests
(`tools/dsv41-hip/zig_build.sh test test-dsv41-hip-host native-hip dsv41-hip-gpu-test`, then
`tools/dsv41-hip/run_gpu_tests.sh`): smoke 14 of 14, ABI 6 of 6, glue 9 of 10. The one failure is expected on this
box: the GPU's timer runs at 100 MHz while ROCm reports 99.81 MHz (the Ryzen AI Halo's timer really runs at 99.81), so
the `clock` check sees about 0.2% more time than the host. The engine uses that rate only for timings and GPU wait
deadlines, so on an M5 they run about 0.2% short.

That's the stack in [Tested on](../README.md#tested-on). From here, link the two boxes with two USB4 cables
([TUNING.md](TUNING.md)) and follow [BUILD_AND_RUN.md](../tools/dsv41-hip/BUILD_AND_RUN.md). Each M5 has two USB4
ports, so a pair uses both.

## If something goes wrong

- **The box doesn't see the USB stick:** Fast Boot is still on, or the stick is in a front port; try a rear USB-A port.
- **It starts Windows instead of the installer:** use Boot Override (step 1) rather than changing the boot order.
- **`DriverUnavailable` from StrixFold's programs:** `therock-gfx1151` is missing, or run `sudo ldconfig` (step 7).
- **The GPU shows about 62 GiB instead of 124:** the options from step 5 aren't in the boot menu. Run
  `sudo update-grub` again and restart.
- **"bad shim signature" or the kernel won't start:** the certificate isn't enrolled. In the GRUB menu choose
  **Advanced options** and the Debian kernel, then repeat step 6.
