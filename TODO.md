# TODO

Go for MIDI creation after UNMIXX stem separations :)

## Three simultaneous lead singers

The current two-output UNMIXX path produces leaky stems for the three-singer
test. Manual alignment can correct source order, but it cannot remove leakage
or keep a stable `single singer` versus `two-singer remainder` grouping.

Possible paths forward:

1. **Reference-conditioned target-singer extraction (preferred).** Obtain a
   clean solo reference clip for each performer, train or fine-tune a
   singing-domain target-extraction model, and run it once per performer. The
   model should use the reference as its target identity and extract that
   singer from the all-vocals mixture; constrain all extracted stems to
   reconstruct the mixture.

2. **Fixed three-output blind singing-voice separator.** Train a new
   UNMIXX-style model with three output sources on mixtures that have three
   clean individual vocal stems. Start with exactly three sources rather than
   an arbitrary source-count argument; add variable-count support only after
   the three-output model separates and tracks the intended singers reliably.
