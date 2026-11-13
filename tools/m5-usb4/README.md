# m5-usb4: the USB4 host-to-host setting on a Bosgame M5

Optional and advanced. What it does to StrixFold (+1–2% decode, +6% prompt reading) and why it is optional are in
[BOSGAME-M5.md](../../docs/BOSGAME-M5.md#optional-the-host-to-host-setting). Read the risk there first: the module
sends a message to the chip's system management unit (SMU), and a wrong one can hang the box until a power cycle.

| File | What it is |
|---|---|
| `bos_usb4h2h.c`, `Makefile` | A Linux kernel module that sends AMD's "USB4 H2H Enhancement" to both USB4 routers through the SMU mailbox the BIOS uses, as Bosgame's BIOS 1.10 does at start-up. GPL-2.0, as kernel modules are (`COPYING`). |
| `m5-usb4` | `status` (each cable's partner and speed, and whether the controllers are kept awake), `awake` (keeps the USB4 controllers and routers out of runtime sleep), `h2h` (applies the setting). Runs only on the AXB35-02 board the M5 is built on, and loads only a module whose SHA-256 you pin in it. |
| `m5-usb4.service` | At every start: `awake`, then `h2h`, before the network comes up. |

Everything below runs on each M5, with the 7.2.7 kernel and Secure Boot set up as in
[BOSGAME-M5.md](../../docs/BOSGAME-M5.md). We apply the setting on both boxes of a pair.

## Build and sign

With Secure Boot on, the kernel loads the module only if it is signed with a key it trusts. Make your own key and enrol
it once, as in [BOSGAME-M5.md, Cooling](../../docs/BOSGAME-M5.md#cooling) (`MOK.priv` and `MOK.der`; keep `MOK.priv`
private). Then, in this folder:

```bash
sudo apt install build-essential linux-headers-7.2.7+rex-amd64
make -C /lib/modules/$(uname -r)/build M=$PWD modules
/usr/src/linux-headers-$(uname -r)/scripts/sign-file sha256 MOK.priv MOK.der bos_usb4h2h.ko
```

A first, harmless check: `test_only=1` sends only the SMU's test message and nothing else.

```bash
sudo insmod bos_usb4h2h.ko test_only=1 && sudo dmesg | grep bos_usb4h2h | tail -1 && sudo rmmod bos_usb4h2h
# bos_usb4h2h: SMU test message: response 0x1, arg0 0x1234 -> 0x1235 (expect 0x1235)
```

Any other answer: stop here.

## Install

Pin the module's SHA-256 in the script, then install the three files root-owned:

```bash
sed -i "s/^SHA=.*/SHA=$(sha256sum bos_usb4h2h.ko | cut -d' ' -f1)/" m5-usb4
sudo install -D -o root -g root -m 644 bos_usb4h2h.ko /usr/local/lib/m5-usb4/bos_usb4h2h.ko
sudo install -o root -g root -m 755 m5-usb4 /usr/local/sbin/m5-usb4
sudo install -o root -g root -m 644 m5-usb4.service /etc/systemd/system/m5-usb4.service
sudo systemctl daemon-reload
```

## Use

Stop the model on the pair first: the cables drop for about 10–20 s.

```bash
sudo m5-usb4 status    # cable 0 and cable 1: partner, speed per lane, lanes
sudo m5-usb4 h2h       # the module's two lines, the two controllers reset, then the speeds again
```

Our numbers are with the links at Gen 3 (20 Gb/s per lane in `status`); see
[The USB4 links](../../docs/BOSGAME-M5.md#the-usb4-links) for bringing them up from Gen 2. To check the result, run
`iperf3 -s` on node0 and `iperf3 -c 10.40.0.1 -t 8` on node1: about 30 Gb/s on one cable with the setting, about 9
without.

A restart clears the setting. To apply it at every start (and keep the controllers awake):

```bash
sudo systemctl enable m5-usb4.service
```

## Undo

Nothing is stored: `sudo systemctl disable m5-usb4.service` and restart. To remove the files:
`sudo rm /usr/local/sbin/m5-usb4 /etc/systemd/system/m5-usb4.service; sudo rm -r /usr/local/lib/m5-usb4`.

## The module's parameters

- `enable=1` (the default) turns the setting on; `enable=0` writes the BIOS's "Disabled"; `enable=-1` leaves it alone.
- `test_only=1` sends only the test message.
- `stapm_mw`, `fast_mw`, `slow_mw` set the power limits in milliwatts with the SMU messages ryzenadj uses (0, the
  default, leaves them alone). `m5-usb4` doesn't use them.
