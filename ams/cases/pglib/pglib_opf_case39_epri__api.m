%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%                                                                  %%%%%
%%%%    IEEE PES Power Grid Library - Optimal Power Flow - v23.07     %%%%%
%%%%          (https://github.com/power-grid-lib/pglib-opf)           %%%%%
%%%%             Benchmark Group - Active Power Increase              %%%%%
%%%%                         23 - July - 2023                         %%%%%
%%%%                                                                  %%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function mpc = pglib_opf_case39_epri__api
mpc.version = '2';
mpc.baseMVA = 100.0;

%% bus data
%	bus_i	type	Pd	Qd	Gs	Bs	area	Vm	Va	baseKV	zone	Vmax	Vmin
mpc.bus = [
	1	 1	 160.69	 44.20	 0.0	 0.0	 2	    1.00000	    0.00000	 345.0	 1	    1.06000	    0.94000;
	2	 1	 0.0	 0.0	 0.0	 0.0	 2	    1.00000	    0.00000	 345.0	 1	    1.06000	    0.94000;
	3	 1	 530.15	 2.40	 0.0	 0.0	 2	    1.00000	    0.00000	 345.0	 1	    1.06000	    0.94000;
	4	 1	 823.22	 184.00	 0.0	 0.0	 1	    1.00000	    0.00000	 345.0	 1	    1.06000	    0.94000;
	5	 1	 0.0	 0.0	 0.0	 0.0	 1	    1.00000	    0.00000	 345.0	 1	    1.06000	    0.94000;
	6	 1	 0.0	 0.0	 0.0	 0.0	 1	    1.00000	    0.00000	 345.0	 1	    1.06000	    0.94000;
	7	 1	 384.94	 84.00	 0.0	 0.0	 1	    1.00000	    0.00000	 345.0	 1	    1.06000	    0.94000;
	8	 1	 859.44	 176.60	 0.0	 0.0	 1	    1.00000	    0.00000	 345.0	 1	    1.06000	    0.94000;
	9	 1	 6.50	 -66.60	 0.0	 0.0	 1	    1.00000	    0.00000	 345.0	 1	    1.06000	    0.94000;
	10	 1	 0.0	 0.0	 0.0	 0.0	 1	    1.00000	    0.00000	 345.0	 1	    1.06000	    0.94000;
	11	 1	 0.0	 0.0	 0.0	 0.0	 1	    1.00000	    0.00000	 345.0	 1	    1.06000	    0.94000;
	12	 1	 14.04	 88.00	 0.0	 0.0	 1	    1.00000	    0.00000	 345.0	 1	    1.06000	    0.94000;
	13	 1	 0.0	 0.0	 0.0	 0.0	 1	    1.00000	    0.00000	 345.0	 1	    1.06000	    0.94000;
	14	 1	 0.0	 0.0	 0.0	 0.0	 1	    1.00000	    0.00000	 345.0	 1	    1.06000	    0.94000;
	15	 1	 526.86	 153.00	 0.0	 0.0	 3	    1.00000	    0.00000	 345.0	 1	    1.06000	    0.94000;
	16	 1	 541.68	 32.30	 0.0	 0.0	 3	    1.00000	    0.00000	 345.0	 1	    1.06000	    0.94000;
	17	 1	 0.0	 0.0	 0.0	 0.0	 2	    1.00000	    0.00000	 345.0	 1	    1.06000	    0.94000;
	18	 1	 260.14	 30.00	 0.0	 0.0	 2	    1.00000	    0.00000	 345.0	 1	    1.06000	    0.94000;
	19	 1	 0.0	 0.0	 0.0	 0.0	 3	    1.00000	    0.00000	 345.0	 1	    1.06000	    0.94000;
	20	 1	 1119.58	 103.00	 0.0	 0.0	 3	    1.00000	    0.00000	 345.0	 1	    1.06000	    0.94000;
	21	 1	 451.12	 115.00	 0.0	 0.0	 3	    1.00000	    0.00000	 345.0	 1	    1.06000	    0.94000;
	22	 1	 0.0	 0.0	 0.0	 0.0	 3	    1.00000	    0.00000	 345.0	 1	    1.06000	    0.94000;
	23	 1	 407.49	 84.60	 0.0	 0.0	 3	    1.00000	    0.00000	 345.0	 1	    1.06000	    0.94000;
	24	 1	 308.60	 -92.20	 0.0	 0.0	 3	    1.00000	    0.00000	 345.0	 1	    1.06000	    0.94000;
	25	 1	 368.80	 47.20	 0.0	 0.0	 2	    1.00000	    0.00000	 345.0	 1	    1.06000	    0.94000;
	26	 1	 228.85	 17.00	 0.0	 0.0	 2	    1.00000	    0.00000	 345.0	 1	    1.06000	    0.94000;
	27	 1	 462.65	 75.50	 0.0	 0.0	 2	    1.00000	    0.00000	 345.0	 1	    1.06000	    0.94000;
	28	 1	 339.17	 27.60	 0.0	 0.0	 3	    1.00000	    0.00000	 345.0	 1	    1.06000	    0.94000;
	29	 1	 466.76	 26.90	 0.0	 0.0	 3	    1.00000	    0.00000	 345.0	 1	    1.06000	    0.94000;
	30	 2	 0.0	 0.0	 0.0	 0.0	 2	    1.00000	    0.00000	 345.0	 1	    1.06000	    0.94000;
	31	 3	 15.15	 4.60	 0.0	 0.0	 1	    1.00000	    0.00000	 345.0	 1	    1.06000	    0.94000;
	32	 2	 0.0	 0.0	 0.0	 0.0	 1	    1.00000	    0.00000	 345.0	 1	    1.06000	    0.94000;
	33	 2	 0.0	 0.0	 0.0	 0.0	 3	    1.00000	    0.00000	 345.0	 1	    1.06000	    0.94000;
	34	 2	 0.0	 0.0	 0.0	 0.0	 3	    1.00000	    0.00000	 345.0	 1	    1.06000	    0.94000;
	35	 2	 0.0	 0.0	 0.0	 0.0	 3	    1.00000	    0.00000	 345.0	 1	    1.06000	    0.94000;
	36	 2	 0.0	 0.0	 0.0	 0.0	 3	    1.00000	    0.00000	 345.0	 1	    1.06000	    0.94000;
	37	 2	 0.0	 0.0	 0.0	 0.0	 2	    1.00000	    0.00000	 345.0	 1	    1.06000	    0.94000;
	38	 2	 0.0	 0.0	 0.0	 0.0	 3	    1.00000	    0.00000	 345.0	 1	    1.06000	    0.94000;
	39	 2	 1817.67	 250.00	 0.0	 0.0	 1	    1.00000	    0.00000	 345.0	 1	    1.06000	    0.94000;
];

%% generator data
%	bus	Pg	Qg	Qmax	Qmin	Vg	mBase	status	Pmax	Pmin
mpc.gen = [
	30	 4.0	 78.2	 400.0	 -243.6	 1.0	 100.0	 1	 8	 0.0; % COW
	31	 755.0	 0.0	 755.0	 -755.0	 1.0	 100.0	 1	 1510	 0.0; % COW
	32	 1277.5	 0.0	 1278.0	 -1278.0	 1.0	 100.0	 1	 2555	 0.0; % COW
	33	 432.5	 0.0	 433.0	 -433.0	 1.0	 100.0	 1	 865	 0.0; % COW
	34	 438.5	 0.0	 439.0	 -439.0	 1.0	 100.0	 1	 877	 0.0; % COW
	35	 560.0	 0.0	 560.0	 -560.0	 1.0	 100.0	 1	 1120	 0.0; % COW
	36	 461.0	 0.0	 461.0	 -461.0	 1.0	 100.0	 1	 922	 0.0; % COW
	37	 759.0	 0.0	 759.0	 -759.0	 1.0	 100.0	 1	 1518	 0.0; % COW
	38	 594.5	 0.0	 595.0	 -595.0	 1.0	 100.0	 1	 1189	 0.0; % COW
	39	 1336.0	 0.0	 1336.0	 -1336.0	 1.0	 100.0	 1	 2672	 0.0; % COW
];

%% generator cost data
%	2	startup	shutdown	n	c(n-1)	...	c0
mpc.gencost = [
	2	 0.0	 0.0	 3	   0.000000	   6.724778	   0.000000; % COW
	2	 0.0	 0.0	 3	   0.000000	  14.707625	   0.000000; % COW
	2	 0.0	 0.0	 3	   0.000000	  24.804734	   0.000000; % COW
	2	 0.0	 0.0	 3	   0.000000	  34.844643	   0.000000; % COW
	2	 0.0	 0.0	 3	   0.000000	  24.652994	   0.000000; % COW
	2	 0.0	 0.0	 3	   0.000000	  32.306483	   0.000000; % COW
	2	 0.0	 0.0	 3	   0.000000	  18.157477	   0.000000; % COW
	2	 0.0	 0.0	 3	   0.000000	  31.550181	   0.000000; % COW
	2	 0.0	 0.0	 3	   0.000000	  22.503168	   0.000000; % COW
	2	 0.0	 0.0	 3	   0.000000	  27.434444	   0.000000; % COW
];

%% branch data
%	fbus	tbus	r	x	b	rateA	rateB	rateC	ratio	angle	status	angmin	angmax
mpc.branch = [
	1	 2	 0.0035	 0.0411	 0.6987	 600.0	 600.0	 600.0	 0.0	 0.0	 1	 -30.0	 30.0;
	1	 39	 0.001	 0.025	 0.75	 1000.0	 1000.0	 1000.0	 0.0	 0.0	 1	 -30.0	 30.0;
	2	 3	 0.0013	 0.0151	 0.2572	 500.0	 500.0	 500.0	 0.0	 0.0	 1	 -30.0	 30.0;
	2	 25	 0.007	 0.0086	 0.146	 500.0	 500.0	 500.0	 0.0	 0.0	 1	 -30.0	 30.0;
	2	 30	 0.0	 0.0181	 0.0	 900.0	 900.0	 2500.0	 1.025	 0.0	 1	 -30.0	 30.0;
	3	 4	 0.0013	 0.0213	 0.2214	 500.0	 500.0	 500.0	 0.0	 0.0	 1	 -30.0	 30.0;
	3	 18	 0.0011	 0.0133	 0.2138	 500.0	 500.0	 500.0	 0.0	 0.0	 1	 -30.0	 30.0;
	4	 5	 0.0008	 0.0128	 0.1342	 600.0	 600.0	 600.0	 0.0	 0.0	 1	 -30.0	 30.0;
	4	 14	 0.0008	 0.0129	 0.1382	 500.0	 500.0	 500.0	 0.0	 0.0	 1	 -30.0	 30.0;
	5	 6	 0.0002	 0.0026	 0.0434	 1200.0	 1200.0	 1200.0	 0.0	 0.0	 1	 -30.0	 30.0;
	5	 8	 0.0008	 0.0112	 0.1476	 900.0	 900.0	 900.0	 0.0	 0.0	 1	 -30.0	 30.0;
	6	 7	 0.0006	 0.0092	 0.113	 900.0	 900.0	 900.0	 0.0	 0.0	 1	 -30.0	 30.0;
	6	 11	 0.0007	 0.0082	 0.1389	 480.0	 480.0	 480.0	 0.0	 0.0	 1	 -30.0	 30.0;
	6	 31	 0.0	 0.025	 0.0	 1800.0	 1800.0	 1800.0	 1.07	 0.0	 1	 -30.0	 30.0;
	7	 8	 0.0004	 0.0046	 0.078	 900.0	 900.0	 900.0	 0.0	 0.0	 1	 -30.0	 30.0;
	8	 9	 0.0023	 0.0363	 0.3804	 900.0	 900.0	 900.0	 0.0	 0.0	 1	 -30.0	 30.0;
	9	 39	 0.001	 0.025	 1.2	 900.0	 900.0	 900.0	 0.0	 0.0	 1	 -30.0	 30.0;
	10	 11	 0.0004	 0.0043	 0.0729	 600.0	 600.0	 600.0	 0.0	 0.0	 1	 -30.0	 30.0;
	10	 13	 0.0004	 0.0043	 0.0729	 600.0	 600.0	 600.0	 0.0	 0.0	 1	 -30.0	 30.0;
	10	 32	 0.0	 0.02	 0.0	 900.0	 900.0	 2500.0	 1.07	 0.0	 1	 -30.0	 30.0;
	12	 11	 0.0016	 0.0435	 0.0	 500.0	 500.0	 500.0	 1.006	 0.0	 1	 -30.0	 30.0;
	12	 13	 0.0016	 0.0435	 0.0	 500.0	 500.0	 500.0	 1.006	 0.0	 1	 -30.0	 30.0;
	13	 14	 0.0009	 0.0101	 0.1723	 600.0	 600.0	 600.0	 0.0	 0.0	 1	 -30.0	 30.0;
	14	 15	 0.0018	 0.0217	 0.366	 600.0	 600.0	 600.0	 0.0	 0.0	 1	 -30.0	 30.0;
	15	 16	 0.0009	 0.0094	 0.171	 600.0	 600.0	 600.0	 0.0	 0.0	 1	 -30.0	 30.0;
	16	 17	 0.0007	 0.0089	 0.1342	 600.0	 600.0	 600.0	 0.0	 0.0	 1	 -30.0	 30.0;
	16	 19	 0.0016	 0.0195	 0.304	 600.0	 600.0	 2500.0	 0.0	 0.0	 1	 -30.0	 30.0;
	16	 21	 0.0008	 0.0135	 0.2548	 600.0	 600.0	 600.0	 0.0	 0.0	 1	 -30.0	 30.0;
	16	 24	 0.0003	 0.0059	 0.068	 600.0	 600.0	 600.0	 0.0	 0.0	 1	 -30.0	 30.0;
	17	 18	 0.0007	 0.0082	 0.1319	 600.0	 600.0	 600.0	 0.0	 0.0	 1	 -30.0	 30.0;
	17	 27	 0.0013	 0.0173	 0.3216	 600.0	 600.0	 600.0	 0.0	 0.0	 1	 -30.0	 30.0;
	19	 20	 0.0007	 0.0138	 0.0	 900.0	 900.0	 2500.0	 1.06	 0.0	 1	 -30.0	 30.0;
	19	 33	 0.0007	 0.0142	 0.0	 900.0	 900.0	 2500.0	 1.07	 0.0	 1	 -30.0	 30.0;
	20	 34	 0.0009	 0.018	 0.0	 900.0	 900.0	 2500.0	 1.009	 0.0	 1	 -30.0	 30.0;
	21	 22	 0.0008	 0.014	 0.2565	 900.0	 900.0	 900.0	 0.0	 0.0	 1	 -30.0	 30.0;
	22	 23	 0.0006	 0.0096	 0.1846	 600.0	 600.0	 600.0	 0.0	 0.0	 1	 -30.0	 30.0;
	22	 35	 0.0	 0.0143	 0.0	 900.0	 900.0	 2500.0	 1.025	 0.0	 1	 -30.0	 30.0;
	23	 24	 0.0022	 0.035	 0.361	 600.0	 600.0	 600.0	 0.0	 0.0	 1	 -30.0	 30.0;
	23	 36	 0.0005	 0.0272	 0.0	 900.0	 900.0	 2500.0	 0.0	 0.0	 1	 -30.0	 30.0;
	25	 26	 0.0032	 0.0323	 0.531	 600.0	 600.0	 600.0	 0.0	 0.0	 1	 -30.0	 30.0;
	25	 37	 0.0006	 0.0232	 0.0	 900.0	 900.0	 2500.0	 1.025	 0.0	 1	 -30.0	 30.0;
	26	 27	 0.0014	 0.0147	 0.2396	 600.0	 600.0	 600.0	 0.0	 0.0	 1	 -30.0	 30.0;
	26	 28	 0.0043	 0.0474	 0.7802	 600.0	 600.0	 600.0	 0.0	 0.0	 1	 -30.0	 30.0;
	26	 29	 0.0057	 0.0625	 1.029	 600.0	 600.0	 600.0	 0.0	 0.0	 1	 -30.0	 30.0;
	28	 29	 0.0014	 0.0151	 0.249	 600.0	 600.0	 600.0	 0.0	 0.0	 1	 -30.0	 30.0;
	29	 38	 0.0008	 0.0156	 0.0	 1200.0	 1200.0	 2500.0	 1.025	 0.0	 1	 -30.0	 30.0;
];

% INFO    : === Translation Options ===
% INFO    : Load Model:                  from file ./pglib_opf_case39_epri.m.api.sol
% INFO    : Gen Active Capacity Model:   stat
% INFO    : Gen Reactive Capacity Model: al50ag
% INFO    : Gen Active Cost Model:       stat
% INFO    : 
% INFO    : === Load Replacement Notes ===
% INFO    : Bus 1	: Pd=97.6, Qd=44.2 -> Pd=160.69, Qd=44.20
% INFO    : Bus 3	: Pd=322.0, Qd=2.4 -> Pd=530.15, Qd=2.40
% INFO    : Bus 4	: Pd=500.0, Qd=184.0 -> Pd=823.22, Qd=184.00
% INFO    : Bus 7	: Pd=233.8, Qd=84.0 -> Pd=384.94, Qd=84.00
% INFO    : Bus 8	: Pd=522.0, Qd=176.6 -> Pd=859.44, Qd=176.60
% INFO    : Bus 9	: Pd=6.5, Qd=-66.6 -> Pd=6.50, Qd=-66.60
% INFO    : Bus 12	: Pd=8.53, Qd=88.0 -> Pd=14.04, Qd=88.00
% INFO    : Bus 15	: Pd=320.0, Qd=153.0 -> Pd=526.86, Qd=153.00
% INFO    : Bus 16	: Pd=329.0, Qd=32.3 -> Pd=541.68, Qd=32.30
% INFO    : Bus 18	: Pd=158.0, Qd=30.0 -> Pd=260.14, Qd=30.00
% INFO    : Bus 20	: Pd=680.0, Qd=103.0 -> Pd=1119.58, Qd=103.00
% INFO    : Bus 21	: Pd=274.0, Qd=115.0 -> Pd=451.12, Qd=115.00
% INFO    : Bus 23	: Pd=247.5, Qd=84.6 -> Pd=407.49, Qd=84.60
% INFO    : Bus 24	: Pd=308.6, Qd=-92.2 -> Pd=308.60, Qd=-92.20
% INFO    : Bus 25	: Pd=224.0, Qd=47.2 -> Pd=368.80, Qd=47.20
% INFO    : Bus 26	: Pd=139.0, Qd=17.0 -> Pd=228.85, Qd=17.00
% INFO    : Bus 27	: Pd=281.0, Qd=75.5 -> Pd=462.65, Qd=75.50
% INFO    : Bus 28	: Pd=206.0, Qd=27.6 -> Pd=339.17, Qd=27.60
% INFO    : Bus 29	: Pd=283.5, Qd=26.9 -> Pd=466.76, Qd=26.90
% INFO    : Bus 31	: Pd=9.2, Qd=4.6 -> Pd=15.15, Qd=4.60
% INFO    : Bus 39	: Pd=1104.0, Qd=250.0 -> Pd=1817.67, Qd=250.00
% INFO    : 
% INFO    : === Generator Setpoint Replacement Notes ===
% INFO    : Gen at bus 30	: Pg=520.0, Qg=270.0 -> Pg=0.0, Qg=203.0
% INFO    : Gen at bus 31	: Pg=323.0, Qg=100.0 -> Pg=1392.0, Qg=629.0
% INFO    : Gen at bus 32	: Pg=362.5, Qg=225.0 -> Pg=793.0, Qg=425.0
% INFO    : Gen at bus 33	: Pg=326.0, Qg=125.0 -> Pg=856.0, Qg=269.0
% INFO    : Gen at bus 34	: Pg=254.0, Qg=83.5 -> Pg=858.0, Qg=235.0
% INFO    : Gen at bus 35	: Pg=343.5, Qg=100.0 -> Pg=863.0, Qg=255.0
% INFO    : Gen at bus 36	: Pg=290.0, Qg=120.0 -> Pg=877.0, Qg=202.0
% INFO    : Gen at bus 37	: Pg=282.0, Qg=125.0 -> Pg=897.0, Qg=69.0
% INFO    : Gen at bus 38	: Pg=432.5, Qg=75.0 -> Pg=1187.0, Qg=170.0
% INFO    : Gen at bus 39	: Pg=550.0, Qg=100.0 -> Pg=2464.0, Qg=143.0
% INFO    : 
% INFO    : === Generator Reactive Capacity Atleast Setpoint Value Notes ===
% INFO    : Gen at bus 30	: Qg 203.0, Qmin 140.0, Qmax 400.0 -> Qmin -243.6, Qmax 400.0
% INFO    : Gen at bus 31	: Qg 629.0, Qmin -100.0, Qmax 300.0 -> Qmin -754.8, Qmax 754.8
% INFO    : Gen at bus 32	: Qg 425.0, Qmin 150.0, Qmax 300.0 -> Qmin -510.0, Qmax 510.0
% INFO    : Gen at bus 33	: Qg 269.0, Qmin 0.0, Qmax 250.0 -> Qmin -322.8, Qmax 322.8
% INFO    : Gen at bus 34	: Qg 235.0, Qmin 0.0, Qmax 167.0 -> Qmin -282.0, Qmax 282.0
% INFO    : Gen at bus 35	: Qg 255.0, Qmin -100.0, Qmax 300.0 -> Qmin -306.0, Qmax 300.0
% INFO    : Gen at bus 36	: Qg 202.0, Qmin 0.0, Qmax 240.0 -> Qmin -242.4, Qmax 240.0
% INFO    : Gen at bus 37	: Qg 69.0, Qmin 0.0, Qmax 250.0 -> Qmin -82.8, Qmax 250.0
% INFO    : Gen at bus 38	: Qg 170.0, Qmin -150.0, Qmax 300.0 -> Qmin -204.0, Qmax 300.0
% INFO    : Gen at bus 39	: Qg 143.0, Qmin -100.0, Qmax 300.0 -> Qmin -171.6, Qmax 300.0
% INFO    : 
% INFO    : === Generator Classification Notes ===
% INFO    : COW    10  -   100.00
% INFO    : 
% INFO    : === Generator Active Capacity Stat Model Notes ===
% INFO    : Gen at bus 30 - COW	: Pg=0.0, Pmax=1040.0 -> Pmax=8   samples: 1
% WARNING : Failed to find a generator capacity within (1392.0-6960.0) after 100 samples, using percent increase model
% INFO    : Gen at bus 31 - COW	: Pg=1392.0, Pmax=646.0 -> Pmax=1510   samples: 100
% INFO    : Gen at bus 32 - COW	: Pg=793.0, Pmax=725.0 -> Pmax=2555   samples: 27
% INFO    : Gen at bus 33 - COW	: Pg=856.0, Pmax=652.0 -> Pmax=865   samples: 44
% INFO    : Gen at bus 34 - COW	: Pg=858.0, Pmax=508.0 -> Pmax=877   samples: 33
% INFO    : Gen at bus 35 - COW	: Pg=863.0, Pmax=687.0 -> Pmax=1120   samples: 15
% INFO    : Gen at bus 36 - COW	: Pg=877.0, Pmax=580.0 -> Pmax=922   samples: 12
% INFO    : Gen at bus 37 - COW	: Pg=897.0, Pmax=564.0 -> Pmax=1518   samples: 39
% INFO    : Gen at bus 38 - COW	: Pg=1187.0, Pmax=865.0 -> Pmax=1189   samples: 57
% WARNING : Failed to find a generator capacity within (2464.0-12320.0) after 100 samples, using percent increase model
% INFO    : Gen at bus 39 - COW	: Pg=2464.0, Pmax=1100.0 -> Pmax=2672   samples: 100
% INFO    : 
% INFO    : === Generator Active Capacity LB Model Notes ===
% INFO    : 
% INFO    : === Generator Reactive Capacity Atleast Max 50 Percent Active Model Notes ===
% INFO    : Gen at bus 31 - COW	: Pmax 1510.0, Qmin -754.8, Qmax 754.8 -> Qmin -755.0, Qmax 755.0
% INFO    : Gen at bus 32 - COW	: Pmax 2555.0, Qmin -510.0, Qmax 510.0 -> Qmin -1278.0, Qmax 1278.0
% INFO    : Gen at bus 33 - COW	: Pmax 865.0, Qmin -322.8, Qmax 322.8 -> Qmin -433.0, Qmax 433.0
% INFO    : Gen at bus 34 - COW	: Pmax 877.0, Qmin -282.0, Qmax 282.0 -> Qmin -439.0, Qmax 439.0
% INFO    : Gen at bus 35 - COW	: Pmax 1120.0, Qmin -306.0, Qmax 300.0 -> Qmin -560.0, Qmax 560.0
% INFO    : Gen at bus 36 - COW	: Pmax 922.0, Qmin -242.4, Qmax 240.0 -> Qmin -461.0, Qmax 461.0
% INFO    : Gen at bus 37 - COW	: Pmax 1518.0, Qmin -82.8, Qmax 250.0 -> Qmin -759.0, Qmax 759.0
% INFO    : Gen at bus 38 - COW	: Pmax 1189.0, Qmin -204.0, Qmax 300.0 -> Qmin -595.0, Qmax 595.0
% INFO    : Gen at bus 39 - COW	: Pmax 2672.0, Qmin -171.6, Qmax 300.0 -> Qmin -1336.0, Qmax 1336.0
% INFO    : 
% INFO    : === Generator Setpoint Replacement Notes ===
% INFO    : Gen at bus 30	: Pg=0.0, Qg=203.0 -> Pg=4.0, Qg=78.2
% INFO    : Gen at bus 30	: Vg=1.0 -> Vg=1.0
% INFO    : Gen at bus 31	: Pg=1392.0, Qg=629.0 -> Pg=755.0, Qg=0.0
% INFO    : Gen at bus 31	: Vg=1.0 -> Vg=1.0
% INFO    : Gen at bus 32	: Pg=793.0, Qg=425.0 -> Pg=1277.5, Qg=0.0
% INFO    : Gen at bus 32	: Vg=1.0 -> Vg=1.0
% INFO    : Gen at bus 33	: Pg=856.0, Qg=269.0 -> Pg=432.5, Qg=0.0
% INFO    : Gen at bus 33	: Vg=1.0 -> Vg=1.0
% INFO    : Gen at bus 34	: Pg=858.0, Qg=235.0 -> Pg=438.5, Qg=0.0
% INFO    : Gen at bus 34	: Vg=1.0 -> Vg=1.0
% INFO    : Gen at bus 35	: Pg=863.0, Qg=255.0 -> Pg=560.0, Qg=0.0
% INFO    : Gen at bus 35	: Vg=1.0 -> Vg=1.0
% INFO    : Gen at bus 36	: Pg=877.0, Qg=202.0 -> Pg=461.0, Qg=0.0
% INFO    : Gen at bus 36	: Vg=1.0 -> Vg=1.0
% INFO    : Gen at bus 37	: Pg=897.0, Qg=69.0 -> Pg=759.0, Qg=0.0
% INFO    : Gen at bus 37	: Vg=1.0 -> Vg=1.0
% INFO    : Gen at bus 38	: Pg=1187.0, Qg=170.0 -> Pg=594.5, Qg=0.0
% INFO    : Gen at bus 38	: Vg=1.0 -> Vg=1.0
% INFO    : Gen at bus 39	: Pg=2464.0, Qg=143.0 -> Pg=1336.0, Qg=0.0
% INFO    : Gen at bus 39	: Vg=1.0 -> Vg=1.0
% INFO    : 
% INFO    : === Writing Matpower Case File Notes ===
