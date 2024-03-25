# Rollback Netcode

:Author: Buck Baskin [@buck@fosstodon.org](https://fosstodon.org/@buck)
:Created: 2024-02-28
:Updated: 2024-03-05
:Parent Design: [designs/runtime.md](../designs/runtime.md)

## Overview

Enable rollback to make network latency disappear

FormaK aims to combine symbolic modeling for fast, efficient system modelling
with code generation to create performant code that is easy to use.

The Five Key Elements the library provides to achieve this user experience are:

1. Python Interface to define models
2. Python implementation of the model and supporting tooling
3. Integration to scikit-learn to leverage the model selection and parameter tuning functions
4. C++ and Python to C++ interoperability for performance
5. C++ interfaces to support a variety of model uses

This design provides an extension to the second of the Five Keys "Python
implementation of the model..." and the fifth of the Five Keys "C++ interfaces
to support a variety of model uses" to support easier to use filters. This work
started with the FormaK runtime.

This design takes the next step to implement “netcode” [1][2][3] for handing
sensor data from sensors with delays (either from the sensor itself or from
networking delays).

[1] Netcode is the term I’m familiar with this from gaming networking, but this may have a different name from signal processing. If you have suggestions for a better name, let me know on Mastodon at [@buck](https://fosstodon.org/@buck) https://words.infil.net/w02-netcode.html
[2] Netcode is not the networking. Instead it's the code to handle signals that are transmitted over an unreliable or delayed network
[3] Another name for this concept is a [backtracking filter, example](https://www.mdpi.com/1424-8220/22/9/3289)

Gaming netcode has a number of strategies for handling networking delays,
packet re-ordering or packet drop in a multiplayer system. The two primary
strategies are delay-based netcode and rollback netcode.

[Delay-based netcode]( https://words.infil.net/w02-netcode-p3.html ) is the
simpler strategy: defer processing until a fixed delay after the message’s
nominal timestamp based on the expected network delay. That way, if other
messages arrive out of order within the time window, they can be re-ordered and
processed correctly. Simplicity is the primary virtue, but in the gaming world
it can lead to poor in-game experience with even small amounts of dropped
packets or delayed packets. If the incoming message exceeds the delay, then the
game has to wait for the input and further delay the local user’s input. In the
filtering world, waiting for messages with fixed delay will translate to the
estimated state output being delayed and common users (such as controllers)
will be adversely impacted by increased delays or variable delays.

[Rollback netcode]( https://words.infil.net/w02-netcode-p4.html ) is a strong
alternative to delay-based netcode. With rollback, the local user input is
performed in sync with the local game simulation. When remote inputs from other
players arrive, the game rolls back to the state of the game when the remote
user’s input occurs, then plays the game forward with both user inputs. In the
gaming case, users are preserving their input for some time and character
actions take non-zero time so the game ends up in the correct state over time
and the local user feels like they’re consistently able to take actions with
known reaction times. For FormaK models, sensors are the “remote players” and
rollback allows for consistently producing the best possible state estimate
while being able to easily accommodate out of order sensor readings. Instead of
having a full game engine running, FormaK can use the motion model to fill in
times with missing sensor readings.

The downside to rollback is additional complication, additional performance
restrictions and increased memory requirements. The rollback decisions, error
handling and loading and saving state all become more complicated than simply
shifting the time horizon in the delay approach. In addition, having to save
multiple states increases memory usage and having to account for the time to
replay through multiple states means that each update must be only a small
fraction of the normal time for updates instead of being able to use 80-100% of
the time budget. Rollback also stresses the state management logic to ensure
that all parts of the state are rolled-back and replayed together, so if state
management isn't done carefully some aspects can become desynced (e.g. if audio
tracks are replayed out of time with game visuals during rollback).

These rollback downsides are in fact an upside opportunity for FormaK. With a
reliable, fast and memory efficient rollback implementation, it will become
easy to adopt either delay-based or rollback-based netcode for filtering
applications. Users of FormaK will not be required to implement this themselves
and instead can jump directly to an efficient sensor management scheme.

### Design Considerations

- Finite horizon (memory based, time based, fixed value?) for rollback history, drop some messages
- Allow fused delay and rollback if a user wants to configure it
- Additional constraints on the system applied due to netcode https://words.infil.net/w02-netcode-p5.html
    - fast state serialization, deserialization
    - separate state update logic from “visualization”/publishing
    - software performance https://www.youtube.com/watch?v=7jb0FOcImdg&t=1570s “Eight Frames in 16ms”
    - updates must be deterministic
- One additional benefit of the state estimation use case is that only one estimator needs to be kept in sync, the sensors don’t need to synchronize
- GGPO as heavy inspiration https://words.infil.net/w02-netcode-p5.html Use some of this quote
    
    > The main function of GGPO is to provide a **robust solution for syncing machines and game states**.
     For example, GGPO can tell your game when two machines are out of sync,
     and by how much. It can also keep track of the inputs from all 
    connected parties and tell your game when to roll back, by how much, and
     what new inputs to apply during your rollbacks.
    > 
    > Because GGPO is well-tested and proven as a solution to handling these 
    > parts of the rollback framework, it is an excellent asset towards making
    >  games that want to use rollback netcode. However, the act of actually **performing the rollbacks still falls on the developers**. They have to make sure their game is performant, their game logic is properly split away from other parts of their game loop, and all edge cases are handled intelligently. GGPO simply tells them when and why to roll back, and what new inputs you should simulate your game with.
    > 
    > … If you are further interested in the exact ways GGPO handles rollback and talks to games, I recommend reading this [[excellent article in Game Developer Magazine](https://drive.google.com/file/d/1nRa3cRBQmKj0-SEyrT_1VNOkPOJWNhVI/view)](https://drive.google.com/file/d/1nRa3cRBQmKj0-SEyrT_1VNOkPOJWNhVI/view)
    >  written by Tony Cannon. It has a bit more of a technical focus, but may
    >  interest you if you’re a programmer or curious on the exact things GGPO
    >  provides your game. You can also peruse the [[source code](https://github.com/pond3r/ggpo)](https://github.com/pond3r/ggpo) and read [[Tony’s own description](https://github.com/pond3r/ggpo/tree/master/doc)](https://github.com/pond3r/ggpo/tree/master/doc) of the theory behind rollback.

### References

- GGPO
    - Source Code: https://github.com/pond3r/ggpo
    - [Author's own description](https://github.com/pond3r/ggpo/tree/master/doc)](https://github.com/pond3r/ggpo/tree/master/doc)
    - [Magazine Writeup](https://drive.google.com/file/d/1nRa3cRBQmKj0-SEyrT_1VNOkPOJWNhVI/view)
- [Fightin' Words](https://words.infil.net/w02-netcode.html)
- Example of a Backtracking Filter https://www.mdpi.com/1424-8220/22/9/3289
 
## Solution Approach

This design will split the implementation between two tasks:
1. Coordination layer for rollback
2. Rolling back and forth the states and sensor readings for the Kalman Filter

This will allow for testing the two separately, and once the coordination is
working, the storage and rolling of the filter should be relatively
straightforward.

### Estimator Interface

The Kalman Filter (or other estimator) only needs to provide the process model
and sensor model (the existing interfaces). 

```cpp
process_model(max_dt, {state, covariance}, calibration, control) -> {state, covariance}

sensor_model(ekf_impl, {state, covariance}, calibration) -> {state, covariance}
```

The load and store functionality for rollback shall be implemented separately.
With this design, the storage and model can be implemented separately and then
composed. 

As a secondary benefit, this design is also compatible with the existing
`ManagedFilter` interface and implementation where the `ManagedFilter` manages
storage (implicitly) and the user provided estimator implements the process
model and sensor model.

In a previous design iteration, the design required the model to also provide
load and save functionality; however, this seemed like an unnecessary burden to
place on the model. As an additional demerit, the previous design iteration
required re-implementing storage for each model.

### Storage Interface

With the load and save functionality offloaded from the model, a Storage class
is needed to provide the load and store functionality.

```cpp
load(time) -> ({state, covariance}, control, sensors)

store(time, {state, covariance}, control, sensors) -> void
```

Note: the controls and calibration inputs are optional. The sensors will always
be a list but the list can be empty.

The user can also override the storage class type as long as it provides the
same interface.

The storage algorithm is unspecified, but the `load` call should return the
latest state that is before the specified time if any are available. Otherwise,
return the earliest state (that will be the equal to or later than the given
time).

The storage algorithm can also provide a time resolution argument. If two times
are within the time resolution, they can be treated as stored at the same time.

With an in-name-only analogy to 
[copy elision](https://en.cppreference.com/w/cpp/language/copy_elision), the
design algorithm has the opportunity to elide loads and stores when it doesn't
impact the outcome of the calculation. For example, if the first sensor from
the next update is after the last stored state, then the algorithm can used the
[memoized](https://en.wikipedia.org/wiki/Memoization) state that's already in
memory instead of calling load. If load is fast enough or this would place a
large computational burden on the algorithm implementation it may be skipped in
the final design.

### Simplification Opportunity

The original `ManagedFilter` behavior is intuitively a subset of the rollback
behavior. For example, if the maximum history is set to 1, then the state will
be set by rolling backwards from the current time.

If this works out to be truly the case, then the `ManagedFilter` class can be
re-implemented as the rollback logic with a default configuration that matches
the old behavior. This would then only require maintaining a single
implementation going forward.

## Feature Tests

1. Coordination layer for rollback
2. Rolling back and forth the states and sensor readings for the Kalman Filter

The first level of feature test is a feature with a simplified model where it
is trivial to inspect that the rollback is working correctly. The model for
this test will use a float to track the time and then a list of strings to
track the order in which sensor readings arrive. The true time for sensor
readings is in 
[lexicographical order](https://en.wikipedia.org/wiki/Lexicographic_order), so
if rollback is working correctly any ordering of the readings should result in
a sorted string for the state at the end of the test. As a secondary test, the
time of the internal state should match the expected state. This will ensure
that state loading is correctly handling times and fully serializing the state.

The second level of feature test is to rerun a Kalman Filter with the rollback
logic specifically. The base case will be a `ManagedFilter` of the original
kind that runs forward in time with the messages perfectly synchronized. The
rollback filter should successfully update the state with randomly shuffled
messages such that the end state looks as if it were perfectly synchronized.

Note: The feature tests will not cover the error handling logic for messages
that arrive so late they violate the time limit, message limit or memory limit.

## Roadmap and Process

1. Write a design
2. Write a feature test(s)
3A. Experiments
3B. Build a simple prototype
4. Pass feature tests
5. Refactor/cleanup
6. Build an instructive prototype (e.g. something that looks like the project vision but doesn't need to be the full thing)
7. Add unit testing, etc
8. Refactor/cleanup
9. Write up successes, retro of what changed (so I can check for this in future designs)

## Post Review

### 2024-03-01

Feature tests updated to be much more specific with their coverage. The
original specification was vague. See the [Feature Tests](#feature-tests)
section for the expanded explanation of testing.

### 2024-03-02

N.B. The design revisions for today show the benefit of feature testing. It has
helped identify multiple missing aspects of the design and missing testing.

Updates to the [Solution Approach](#solution-approach) to document the
[estimator interface](#estimator-interface) and 
[storage interface](#storage-interface).

This is also the time where I realized that I'd missed the controls inputs.
The calibration doesn't need to be stored and loaded because it doesn't change
over time.

As an implementation note, the rollback will be implemented separated from the
`ManagedFilter` and then merged later. This will ease testing and allow for
pre-confirming that the equivalent behavior can be maintained by selecting
rollback options.
