# Cases

This folder contains various power system case studies used for scheduling studies. Each subfolder holds a different power system model with multiple variants.

## 5bus

PJM 5-bus system [5bus](#references)

- `pjm5bus_demo.xlsx`: Demo case for the 5-bus system.
- `pjm5bus_jumper.xlsx`: Jumper case for the 5-bus system.
- `pjm5bus_uced.json`: UCED case in JSON format.
- `pjm5bus_uced.xlsx`: UCED case in Excel format.
- `pjm5bus_uced_esd1.xlsx`: UCED case with energy storage device.
- `pjm5bus_uced_ev.xlsx`: UCED case with electric vehicles.

## ieee14

IEEE 14-bus system [pstca](#references)

- `ieee14.json`: JSON format of the IEEE 14-bus system.
- `ieee14.raw`: Raw power flow data for the IEEE 14-bus system.
- `ieee14_uced.xlsx`: UCED case for the IEEE 14-bus system.

## ieee39

IEEE 39-bus system [pstca](#references)

- `ieee39.xlsx`: Base case for the IEEE 39-bus system.
- `ieee39_uced.xlsx`: UCED case for the IEEE 39-bus system.
- `ieee39_uced_esd1.xlsx`: UCED case with energy storage device.
- `ieee39_uced_pvd1.xlsx`: UCED case with photovoltaic device.
- `ieee39_uced_vis.xlsx`: Visualization case for the IEEE 39-bus system.

## ieee123

IEEE 123-bus system [pstca](#references)

- `ieee123.xlsx`: Base case for the IEEE 123-bus system.
- `ieee123_regcv1.xlsx`: Case with regulator control version 1.

## matpower

Cases from Matpower [matpower](#references)

- `case5.m`: Matpower case for the 5-bus system.
- `case14.m`: Matpower case for the 14-bus system.
- `case39.m`: Matpower case for the 39-bus system.
- `case118.m`: Matpower case for the 118-bus system.
- `case300.m`: Matpower case for the 300-bus system.
- `case_ACTIVSg2000.m`: Matpower case for the ACTIVSg2000 system.
- `benchmark.json`: Benchmark results, NOT a power system case.

## npcc

Northeast Power Coordinating Council system [SciData](#references)

- `npcc.m`: Matpower case for the NPCC system.
- `npcc_uced.xlsx`: UCED case for the NPCC system.

## wecc

Western Electricity Coordinating Council system [SciData](#references)

- `wecc.m`: Matpower case for the WECC system.
- `wecc_uced.xlsx`: UCED case for the WECC system.

## hawaii

Hawaii Synthetic Grid – 37 Buses [synthetic](#references), source from [Hawaii Synthetic Grid – 37 Buses](https://electricgrids.engr.tamu.edu/hawaii40/)

- `Hawaii40.m`: Matpower case for the synthetic Hawaii system.
- `Hawaii40.AUX`: [Auxiliary File Format](https://www.powerworld.com/knowledge-base/auxiliary-file-format-10), can contain power flow, dynamics, contingencies, economic studies, etc. data.

## pglib

Cases from the Power Grid Lib - Optimal Power Flow [pglib](#references)

- `pglib_opf_case39_epri__api.m`: PGLib case for the 39-bus system.

---

## References

[5bus]: F. Li and R. Bo, "Small test systems for power system economic studies," IEEE PES General Meeting, 2010, pp. 1-4, doi: 10.1109/PES.2010.5589973.

[pstca]: “Power Systems Test Case Archive - UWEE,” labs.ece.uw.edu. https://labs.ece.uw.edu/pstca/

[matpower]: R. D. Zimmerman, C. E. Murillo-Sanchez, and R. J. Thomas, "MATPOWER: Steady-State Operations, Planning and Analysis Tools for Power Systems Research and Education," Power Systems, IEEE Transactions on, vol. 26, no. 1, pp. 12-19, Feb. 2011. doi: 10.1109/TPWRS.2010.2051168

[SciData]: Q. Zhang and F. Li, “A Dataset for Electricity Market Studies on Western and Northeastern Power Grids in the United States,” Scientific Data, vol. 10, no. 1, p. 646, Sep. 2023, doi: 10.1038/s41597-023-02448-w.

[pglib]: S. Babaeinejadsarookolaee et al., “The Power Grid Library for Benchmarking AC Optimal Power Flow Algorithms,” arXiv.org, 2019. https://arxiv.org/abs/1908.02788

[synthetic] A. B. Birchfield, T. Xu, K. M. Gegner, K. S. Shetye and T. J. Overbye, "Grid Structural Characteristics as Validation Criteria for Synthetic Networks," in IEEE Transactions on Power Systems, vol. 32, no. 4, pp. 3258-3265, July 2017, doi: 10.1109/TPWRS.2016.2616385.
