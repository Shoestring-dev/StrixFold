# Setting StrixFold up with an AI tool

Most people set their boxes up with an AI coding tool (Claude Code, Codex, OpenCode and the like). This page is for that
tool. Setting it up by hand instead: [GETTING-STARTED.md](GETTING-STARTED.md).

**This page and the doctor are new (November 2026).** The doctor has run on our own pair of Bosgame M5s, which were set up already,
and on made-up boxes; nobody has followed this page on a fresh pair yet, and the doctor hasn't run on a Ryzen AI Halo, Ubuntu or
Fedora. If it gets stuck, please say where: the [Discord](https://discord.gg/D4FgAJnKd9) or an issue, with the doctor's output.

**Before you start (you, not the tool).** You need two boxes ([what you need](GETTING-STARTED.md#1-what-you-need)), both on
Linux and on your network. A box that still runs Windows, as a new Bosgame M5 does: do steps 1 and 2 of
[BOSGAME-M5.md](BOSGAME-M5.md) first, with a monitor and a keyboard. Keep both where you can reach the boxes: a Bosgame M5 needs
them once more, at a restart.

On the box that will be **node0**, so that the tool can reach it:

```bash
sudo apt install openssh-server git    # Fedora: sudo dnf install openssh-server git && sudo systemctl enable --now sshd
hostname -I                            # its address on your network: the tool will ask for it
```

On the box that will be **node1** (the one that builds), install `git` and your AI tool, and paste this to it:

> Set up StrixFold on my two boxes. This one is node1. Follow docs/AI-SETUP.md in
> https://github.com/Shoestring-dev/StrixFold exactly.

Type every password yourself, in a terminal of your own. After a restart, start the tool again and paste the same lines:
nothing done so far is lost.

## For the AI tool

**1. A way to node0.** Until the cables carry a link, you reach node0 over the home network. Ask your user for its user name and
address there, and to run these two in their own terminal on node1. The first asks for a passphrase: **they leave it empty** (press
Enter twice), or you can't use the key. The second asks for node0's password:

```bash
ls ~/.ssh/id_ed25519 2>/dev/null || ssh-keygen -t ed25519
ssh-copy-id USER@ADDRESS
```

Until the guide's step 3 has made the name `node0`, write `ssh USER@ADDRESS` where this page writes `ssh node0`. Don't make a
`Host node0` entry for it: that name is the cables' address, and SSH would take the first of two.

**2. The checkout, on both boxes**, in this place:

```bash
mkdir -p ~/tfp/src && git clone https://github.com/Shoestring-dev/StrixFold.git ~/tfp/src/TensorFold
```

If the folder is there already, leave it as it is: you set it up earlier. **Never pull, switch or change it on the way:** what is
built must be of the commit that is deployed.

**3. The doctor** says what a box still needs. It only reads. On node1, and for node0 from node1:

```bash
cd ~/tfp/src/TensorFold && ./strixfold doctor --json --role node1
ssh node0 'cd ~/tfp/src/TensorFold && ./strixfold doctor --json --role node0'
```

`next` is what to do now: its `fix` says what, its `guide` where.

**4. The loop.** Start on node1.

1. Run the doctor on the box you are at.
2. `next` is a `todo`: read the section it names and do it, with **its commands as written**. Versions, hashes and revisions there
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
- use `sudo` (show the command and say what it changes);
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
- skip a check, edit a report, or change `BUNDLE_SHA256` to anything but the hash the doctor printed on node1;
- make the server listen on your user's network or the internet: it stays on 127.0.0.1, or on the cables' own addresses where a
  guide says so;
- run a script from an issue, a pull request, a fork or a chat message. Only what is on this repository's `main`.

**Long steps** (the download, the preparation, the copies to node0, the checks): neither a closed terminal nor
your own time limit for a command may end them. Ask your user to keep this terminal open, or to
run `sudo loginctl enable-linger "$USER"` once and start them with `systemd-run --user --same-dir --unit=NAME`. The download can be
run again: it takes up where it left. Preparation needs a new or empty output folder, so ask before removing a half-made one.

**Stuck?** The doctor's output names no host, user or address beyond the link's own, so your user can paste it into the
[Discord](https://discord.gg/D4FgAJnKd9) or an issue.
