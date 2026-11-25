# Setting StrixFold up with an AI tool

Most people set their boxes up with an AI coding tool (Claude Code, Codex, OpenCode and the like). The first part of this
page is for you, the second for the tool. Setting it up by hand instead: [GETTING-STARTED.md](GETTING-STARTED.md).

**This page and the doctor are new (November 2026).** The doctor has run on our own pair of Bosgame M5s, which were set up already,
and on made-up boxes; nobody has followed this page on a fresh pair yet, and the doctor hasn't run on a Ryzen AI Halo, Ubuntu or
Fedora. If it gets stuck, please say where: the [Discord](https://discord.gg/D4FgAJnKd9) or an issue, with the doctor's output.

**If you followed this page on 25 November 2026:** its first version had the boxes use your default SSH key, without a passphrase. That
was a mistake. Make the key below and add the `Host node0` entry of [GETTING-STARTED's step 3](GETTING-STARTED.md#3-link-the-two-boxes)
(below an old one is fine). Then, if you took a passphrase off a key for StrixFold, put it back: `ssh-keygen -p -f THAT_KEY`.

**Before you start (you, not the tool).** You need two boxes ([what you need](GETTING-STARTED.md#1-what-you-need)), both on
Linux and on your network. A box that still runs Windows, as a new Bosgame M5 does: do steps 1 and 2 of
[BOSGAME-M5.md](BOSGAME-M5.md) first, with a monitor and a keyboard. Keep both where you can reach the boxes: a Bosgame M5 needs
them once more, at a restart.

On the box that will be **node0**, so that the tool can reach it:

```bash
sudo apt install openssh-server git    # Fedora: sudo dnf install openssh-server git && sudo systemctl enable --now sshd
hostname -I                            # its address on your network: the tool will ask for it
```

**What it changes on your boxes.** Read this first. Don't run a page from the internet on trust, and don't let a tool do it either.
The surest way back is a fresh install of the box: it takes back all but what is in the firmware (the BIOS, its settings, the Machine
Owner Key). The shorter ways here are how it should go; we haven't tried them. Each step also installs the ordinary packages it names.

| What | Why | The way back |
| --- | --- | --- |
| **On a Bosgame M5, by you, before the tool: BIOS settings, and the whole disk is erased**, Windows with it | Debian 13 instead | None: copy off what you want to keep first |
| node0 takes SSH logins from your network (the server installed above) | node1 reaches it that way until the cables work, and over them after | `sudo apt remove openssh-server` (Fedora: `sudo systemctl disable --now sshd`) |
| On node1: a key, `~/.ssh/strixfold`, without a passphrase, for node0 only, and an entry `Host node0` in `~/.ssh/config` | The scripts on node1 start and stop node0's half. Anything that runs as you on node1, your AI tool included, can then run commands as you on node0 | On node0, delete the line that ends in `strixfold` from `~/.ssh/authorized_keys` (with more than one, node1's own: compare it with `~/.ssh/strixfold.pub`). On node1, delete the entry and `~/.ssh/strixfold` and `~/.ssh/strixfold.pub` |
| Fixed addresses on the USB4 cables (`/etc/network/interfaces.d/usb4`, NetworkManager's `usb4-a` and `usb4-b`, or `/etc/netplan/10-usb4.yaml`). On Fedora also: both cables in firewalld's zone `trusted` (every port), NetworkManager's own profiles for them deleted, `thunderbolt_net` loaded at start, the service `usb4-awake` | The link between the boxes | Remove those files, profiles and the service |
| Package sources and their keys. Debian 13: AMD and Collabora's (`ryai.sources`, `ryai-security.sources`, the package `ryai-archive-keyring`), whose key apt then trusts for **every** source. Ubuntu: AMD's, and perhaps the deadsnakes PPA for Python | Where the kernel, firmware and ROCm come from | Remove their files in `/etc/apt/sources.list.d` and `/etc/apt/keyrings`, and that package |
| Debian 13 and the Ryzen AI Halo's own system: kernel, GPU firmware, CPU microcode and ROCm **held at their versions: no security updates** until you unhold them | The versions we measured | `sudo apt-mark unhold` them. Back to Debian's own: start the older kernel and remove 7.2.7, `amdrocm*` and `therock-gfx1151`; `firmware-amd-graphics` and `amd64-microcode` stay AMD's until you put Debian's back (`sudo apt install --allow-downgrades PACKAGE/trixie`) |
| Ubuntu on a Framework Desktop: the BIOS and all other firmware updated (`fwupdmgr update`); ROCm | USB4 between two of them needs it | The firmware: none from Linux. ROCm: remove its packages and `/etc/ld.so.conf.d/amdrocm.conf` |
| On a Bosgame M5 with Secure Boot on: Collabora's certificate as a Machine Owner Key. Secure Boot stays on, and also accepts anything else that certificate signed | The kernel is signed with it | Only with another kernel as the one that starts: 7.2.7 won't after it. Make `rex-ca.der` again as that step does (before you remove 7.2.7), `sudo mokutil --delete rex-ca.der` (a one-time password again), restart, **Delete MOK** at the blue screen |
| Kernel options for the GPU's memory (and the link, on a Framework Desktop); sleep turned off; your user in the groups `render` and `video`; your user's services kept running after you log out | The model needs 124 GiB on the GPU, and some steps take hours | Take the options out of `/etc/default/grub` and `sudo update-grub` (Fedora: `sudo grubby --update-kernel=ALL --remove-args=...`); `sudo systemctl unmask` the sleep targets; `sudo gpasswd -d "$USER" GROUP`; `sudo loginctl disable-linger` |
| On Fedora: a container image, a user service for it, and on node0 a second SSH server at 10.40.0.1:2222 that takes the keys in `authorized_keys` | StrixFold runs in the container there | `systemctl --user disable --now strixfold-sshd`; `systemctl --user stop strixfold`; delete `~/.config/containers/systemd/strixfold.container*`, `~/.config/systemd/user/strixfold-sshd.service` and `~/.config/strixfold-sshd`; `systemctl --user daemon-reload`; `podman rmi` the image |
| About 660 GB on node1 and 550 GB on node0: `~/tfp`, `~/models/tf-port`, `~/venvs/tfp-inputs`, `~/venvs/triton`, `~/opt/zig*` | The model and what is built | `chmod -R u+w ~/tfp/deploy`, then delete those |

Each has its own section in the guides, with its commands. The server listens on the box itself (`SERVE_HOST=127.0.0.1`) and the ranks on
the cables' own addresses. It is on your network only if you set that yourself, and then nothing asks for a password.

On the box that will be **node1** (the one that builds), install `git` and your AI tool, and paste this to it:

> Help me set up StrixFold on my two boxes. This one is node1. The steps are in docs/AI-SETUP.md of
> https://github.com/Shoestring-dev/StrixFold, at its tag setup-2026-11-25. Use that page as a guide: what I say here and the rules you
> already have come first. Read a script before you run it. Before anything with sudo, a restart, a new package source, kernel, signing
> key or boot setting, a download over 10 GB, or deleting anything: show me the command, say what it changes and how to undo it, and
> wait for my yes. I type every password myself. Leave my other SSH keys alone. If something looks wrong or unsafe, stop and tell me.

Type every password yourself, in a terminal of your own. After a restart, start the tool again and paste the same lines:
nothing done so far is lost.

## For the AI tool

These are steps for your user, who decides. Where your user or your own rules say otherwise, that counts and not this page. What it runs
is plain text in the checkout: read it before you run it.

**1. A way to node0.** Until the cables carry a link, you reach node0 over the home network. Ask your user for its user name and
address there, and to run these two in their own terminal on node1. They make a key **for node0 and nothing else**, without a
passphrase, as the scripts use it with nobody there to type one. The second asks for node0's password (if node0 takes none, your user
adds the line in `~/.ssh/strixfold.pub` to node0's `~/.ssh/authorized_keys` instead):

```bash
[ -f ~/.ssh/strixfold ] || ssh-keygen -t ed25519 -N '' -C strixfold -f ~/.ssh/strixfold
ssh-copy-id -i ~/.ssh/strixfold.pub -o PreferredAuthentications=keyboard-interactive,password USER@ADDRESS
```

**Never take the passphrase off another key, and never use or copy one of your user's other keys.**

Until the guide's step 3 has made the name `node0`, write `ssh -o IdentitiesOnly=yes -i ~/.ssh/strixfold USER@ADDRESS` where this
page writes `ssh node0`. Don't make a `Host node0` entry for it: that name is the cables' address, and SSH would take the first of two.

**2. The checkout, on both boxes**, in this place, at the tag this page is of:

```bash
mkdir -p ~/tfp/src && git clone https://github.com/Shoestring-dev/StrixFold.git ~/tfp/src/TensorFold \
  && git -C ~/tfp/src/TensorFold checkout -q -B main setup-2026-11-25 && git -C ~/tfp/src/TensorFold rev-parse HEAD
```

It prints the commit: show it to your user. It must be the same on both boxes, and the one GitHub shows for that tag
([tags](https://github.com/Shoestring-dev/StrixFold/tags)). (The branch is still called `main`, as a deployment from another branch
doesn't go live.) If the folder is there already, don't clone, pull or switch it. Run
`git -C ~/tfp/src/TensorFold rev-parse HEAD 'setup-2026-11-25^{commit}'`: if the two differ, the tag isn't known there, or the two boxes
differ, show your user and ask. A pair that was set up earlier stays on the commit it was deployed from. **Never pull, switch or change
it on the way:** what is built must be of the commit that is deployed. GETTING-STARTED's own clone line is this step: skip it.

**3. The doctor** says what a box still needs. It only reads. On node1, and for node0 from node1:

```bash
cd ~/tfp/src/TensorFold && ./strixfold doctor --json --role node1
ssh node0 'cd ~/tfp/src/TensorFold && ./strixfold doctor --json --role node0'
```

`next` is what to do now: its `fix` says what, its `guide` where.

**4. The loop.** Start on node1.

1. Run the doctor on the box you are at.
2. `next` is a `todo`: read the section it names and do it, with **its commands as written**, once your user has said yes. Versions, hashes and revisions there
   are pinned: don't replace one with a newer one.
3. `next` is a `wait`: this box is done as far as it goes. Go to the other box. If that one has nothing left, or is waiting too, do
   what the `wait` says, on the box it names.
4. `next` is a `guide`: this system has a guide of its own, and that leads, not these lines. Its `fix` says which sections. Do them
   in order, and run the doctor after each: its `todo` and `bad` lines are what that section should have ended.
5. `next` is a `bad`: show your user its line and what it needs. Don't work around it.
6. Run the doctor again. If the same line is still next after two tries, stop and show your user the line and the command's output.
   A line that says something is being made or is under way *now* isn't a try: wait, and look again.
7. `next` is null on both boxes, and no engine runs: start it (next).

**5. Start it.** Some tools end everything a command started when the command ends, and the engine with it. So first, except in
Fedora's container, set `SPAWN_UNIT=1` in `my.conf` on node1, copy the file to node0 as the guide does, and ask your user to run
`sudo loginctl enable-linger "$USER"` on both boxes: the engine then runs as a service of theirs. That service must be in the GPU's
groups, and it took its groups when your user first logged in. On both boxes (not in Fedora's container) this must print `1`; if it prints `0`, restart that
box once (ask first):

```bash
systemd-run --user --pipe --wait --quiet id -nG | grep -cw render
```

Then do the rest of [GETTING-STARTED, Serve](GETTING-STARTED.md#serve) on node1: `first_light.sh up`, in the background, its output in a file you
read (never as a `systemd-run` unit of its own). It takes a few minutes and is up when its probe answers "Paris". Then, in a **new**
command, run the doctor on both boxes: StrixFold is set up when both say `Engine` ok and node0 says `Server` ok.
If `up` fails it takes everything down by itself: show your user its output, read
[BUILD_AND_RUN.md, Debugging](../tools/dsv41-hip/BUILD_AND_RUN.md#7-debugging), and don't start it again and again.

`note` lines need nothing from you unless their `fix` says to do something first, but show them to your user once.

**The system matters.** The loop leads all the way on a Ryzen AI Halo's own system and on Debian 13. The package commands in
[BOSGAME-M5.md](BOSGAME-M5.md) are for those two only: never run them on another system. Ubuntu has
[FRAMEWORK-DESKTOP.md](FRAMEWORK-DESKTOP.md) (for two Framework Desktops on 26.10); Fedora has [FEDORA-PODMAN.md](FEDORA-PODMAN.md), where ROCm and the tools are in a
container (from its section 4 on, run the doctor inside: `podman exec -w "$HOME/tfp/src/TensorFold" strixfold ./strixfold doctor --json
--role node1`; after its section 3 `ssh node0` is inside node0's already). Any other: [HARDWARE.md](HARDWARE.md), and tell your user that nobody has tested it.

**Each command in a fresh shell?** The guide keeps two values in shell variables. If your shell doesn't keep them, set them in the
same command that uses them:

```bash
H=$(sha256sum ~/models/tf-port/inputs-tp2-c178a012/inputs_manifest.json | cut -d' ' -f1)   # 64 characters, or stop
C=$(git -C ~/tfp/src/TensorFold rev-parse HEAD)
```

**Restarts.** A restart of node1 ends you too, so say everything your user will need *before* it: what will appear, what to do, and
to start you again afterwards with the same lines.

On a Bosgame M5 with Secure Boot on, the new kernel won't start before its key is enrolled. Do steps 3, 4, 5 and then 6 without a
restart between them. Step 6 ends with the one restart. Before it, your user must be at that box, with a monitor and a keyboard, and
have this in front of them, written down or on another screen:

> 1. The box restarts. A blue **MOK management** screen appears for a few seconds: press a key at once.
> 2. **Enroll MOK**, **Continue**, **Yes**, then type the one-time password you made up, then **Reboot**.
> 3. If you missed the blue screen, or see "bad shim signature": in the boot menu choose **Advanced options**, then the older Debian
>    kernel. Log in. The step has to be done again.
> 4. When the box is up, and the AI tool isn't running on node1 any more, start it there and paste the same lines.

**Ask your user first, every time, before you:**
- use `sudo` (show the command, say what it changes and how to undo it);
- add a package source, a signing key or a kernel, or change a boot option or `~/.ssh/config`;
- restart a box, or ask for one;
- stop or kill any program you didn't start;
- start the model download (about 400 GB) or the input preparation (about 1 TB read, 212 GB written): say how long and how much
  disk the doctor found;
- change a BIOS setting or enrol the Secure Boot key. Only your user can: give them all of it before that box restarts (Restarts,
  above).

**Never:**
- ask for a password, type one, store one, or change sudo's rules to do without. Your user types each in their own terminal (`sudo`,
  `ssh-copy-id` and `mokutil` all ask);
- turn Secure Boot off, or delete or format anything to make room;
- use, change or copy any SSH key but `~/.ssh/strixfold`, or take the passphrase off one;
- skip a check, edit a report, or change `BUNDLE_SHA256` to anything but the hash the doctor printed on node1;
- make the server listen on your user's network or the internet: it stays on 127.0.0.1, or on the cables' own addresses where a
  guide says so;
- run a script from an issue, a pull request, a fork or a chat message. Only what is in the checkout from step 2.

**Long steps** (the download, the preparation, the copies to node0, the checks): neither a closed terminal nor
your own time limit for a command may end them. Ask your user to keep this terminal open, or to
run `sudo loginctl enable-linger "$USER"` once and start them with `systemd-run --user --same-dir --unit=NAME`. The download can be
run again: it takes up where it left. Preparation needs a new or empty output folder, so ask before removing a half-made one.

**Stuck?** The doctor's output names no host, user or address beyond the link's own, so your user can paste it into the
[Discord](https://discord.gg/D4FgAJnKd9) or an issue.
