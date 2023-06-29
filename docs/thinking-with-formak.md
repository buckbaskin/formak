# Thinking With FormaK

Before getting started with FormaK, it can be helpful to know some of the
concepts that will come up and how FormaK thinks about the problem space.

FormaK helps take a model from concept stage to production. This is done by
taking the model through different stages of development.
1. Model definition - detailed model of features
2. Model optimization - fit against data to select parameters
3. Model compilation - compile to Python or C++
4. Model calibration
5. Model runtime

For an Model, there are 4 inter-related concepts at play:
- Model Definition: How does the state evolve over time?
- State: estimated at runtime based on incoming sensor information
- Calibration: provided once at the start of runtime
- Control: provided as truth during runtime as the state evolves over time

An EKF adds:
- Process Model definition (see Model Definition)
	- Process Noise: How much variability do we expect around our control input?
- Sensor Models: How does the state relate to incoming sensor data?
	- Sensor Noise: How much variability do we expect for the incoming sensor data?

How do these relate to each other?
- A State can be calculated online or set to a pre-determined parameter as a Calibration
- A Control can be provided online or set to a pre-determined parameter as a Calibration
- A Control can not be used as part of a sensor model. If you want to use a Control as a sensor model, it should be added to the State and the process model sets the State equal to the Control

Note: Usually these will be referred to as a state vector or a control vector;
however, in FormaK the exact representation can be changed under the hood so
the State, Control, etc are just sets of symbols in a common concept
collection. Examples of internal representation changes include: re-ordering
the states, representing the states in a non-vector format, augmenting the
state vector or simplifying the state vector.
- If you want to access part or all of the State at runtime, define a sensor model to return that state member. This will allow you to access the state regardless of if the underlying state representation changes.
