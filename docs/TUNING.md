# Tuning the two boxes

What we changed on two Ryzen AI Max+ 395 boxes linked by two USB4 cables, and what each change measured. The engine's
settings named here are in `tools/dsv41-hip/serve.conf`.

## The link

- **Two cables, two jobs.** `thunderbolt0` carries the engine's control and per-token traffic, and `thunderbolt1` the
  bulk copies and the stripe. With bulk copies on the per-token cable, a reply's p95 step time rose to 41–165 ms, with
  one at 2.4 s.
- **MTU 1500, not the 65520 maximum.** On thunderbolt_net the big MTU lost packets under bulk load:

  | MTU | One-way bulk | Packets missed |
  |---:|---:|---:|
  | 65520 | 28.0 Gb/s | 4.7% |
  | 9000 | 30.9 Gb/s | 0.22% |
  | 1500 | 31.0 Gb/s | 0.04% |

  At MTU 1500 a striped 2,048-row prompt chunk went from 942 ms to 512 ms, and DwarfStar's prompt reading rose 13%
  (from 264 and 255 tok/s to 292 and 296 at 8,159 tokens). Set it on both boxes' network profiles so it survives a
  reboot.
- **Split the traffic over both cables.** A second proxy thread and a 16 KiB stripe (`COLL_PROXY_STRIPE_KIB=16`) gave
  one chat +6.6% and four chats +8.9%, with identical output.
- **Both directions share the bandwidth.** Expect about 44.5 Gb/s in total, both directions together, not twice the
  one-way number.

## TCP retransmits

The engine's small per-token messages need a short retransmit floor, so a lost packet doesn't stall a reply. With the
same short floor (`TCP_RTO_MIN_US=8000`) on a socket carrying 42 MB prompt exchanges, three of five 8K prompt reads
stalled: the receiver ended 3–7 MB short and hit its deadline. The fix is a short floor for the small messages
(`COLL_RTO_MIN_US=8000`) and a 200 ms floor during bulk phases.

## Power

The boxes serve at the default balanced power setting. The CPU's performance setting is faster: DwarfStar went from
13.3 to 14.7 tok/s. It costs more power, so it is off by default here.

## Several chats

With drafting on, identical prompts decode in lockstep, and leftover drafts break that. Turning drafting off from six
chats at once (`DRAFT_BYPASS_AT=6`) raised eight chats from 97.8 to 108 tok/s, with the output unchanged.
