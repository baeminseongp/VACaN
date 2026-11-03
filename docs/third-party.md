# Third-party provenance

The active mandatory stack is:

- NVIDIA Isaac Sim 5.1.0 and its official Office, Hospital, Warehouse, People,
  and Nova Carter assets, subject to NVIDIA's licenses.
- Isaac Lab 2.3.2, present at `/home/engineer/IsaacLab`.
- Pegasus Simulator 5.1.0, commit
  `644da37e9d5268e5f9a34e78bdcfd57a8bab82b4`, including its `Person` and
  `PersonController` APIs.

VACaN stores asset references, not copies of NVIDIA USD assets. Runtime asset
URLs are resolved with the installed Isaac Sim API.

The repository also retains historical code informed by iCrowdNav
(`BRoln7/icrowdnav`, inspected at commit
`5dc3eda6db8b0e23f790eb215aae8de27e07b744`, MIT License). It is not the active
simulator, robot, pedestrian framework, or Stage 0 compatibility baseline.
