# Rollback Netcode

:Author: Buck Baskin [@buck@fosstodon.org](https://fosstodon.org/@buck)
:Created: 2024-02-28
:Updated: 2024-02-28
:Parent Design: [designs/runtime.md](../designs/runtime.md)

# Overview

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
management isn’t done carefully some aspects can become desynced (e.g. if audio
tracks are replayed out of time with game visuals during rollback).

These rollback downsides are in fact an upside opportunity for FormaK. With a
reliable, fast and memory efficient rollback implementation, it will become
easy to adopt either delay-based or rollback-based netcode for filtering
applications. Users of FormaK will not be required to implement this themselves
and instead can jump directly to an efficient sensor management scheme.

## Design Considerations

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
 
## Solution Approach

This design will split the implementation between two tasks:
1. Coordination layer for rollback
2. Rolling back and forth the states and sensor readings for the Kalman Filter

This will allow for testing the two separately, and once the coordination is
working, the storage and rolling of the filter should be relatively
straghtforward.

## References

- GGPO
    - Source Code: https://github.com/pond3r/ggpo
    - [Author's own description](https://github.com/pond3r/ggpo/tree/master/doc)](https://github.com/pond3r/ggpo/tree/master/doc)
    - [Magazine Writeup](https://drive.google.com/file/d/1nRa3cRBQmKj0-SEyrT_1VNOkPOJWNhVI/view)
- [Fightin' Words](https://words.infil.net/w02-netcode.html)
- Example of a Backtracking Filter https://www.mdpi.com/1424-8220/22/9/3289

## Feature Tests

1. Coordination layer for rollback
2. Rolling back and forth the states and sensor readings for the Kalman Filter

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