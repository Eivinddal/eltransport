function mpc = System
%CASE4GS  Power flow data for 4 bus, 2 gen case from Grainger & Stevenson.
%   Please see CASEFORMAT for details on the case file format.
%
%   This is the 4 bus example from pp. 337-338 of "Power System Analysis",
%   by John Grainger, Jr., William Stevenson, McGraw-Hill, 1994.

%   MATPOWER
Bus_data = xlsread('FilDataTelemarksnett.xlsx','BusData');
Branch_data = xlsread('FilDataTelemarksnett.xlsx','BranchData');


%% MATPOWER Case Format : Version 2
mpc.version = '2';

%%-----  Power Flow Data  -----%%
%% system MVA base
mpc.baseMVA = Bus_data(1,11);

%% bus data

Bus = zeros(length(Bus_data(:,1)),13);
for i = 1:length(Bus_data(:,1))
    Bus(i,1) = round(Bus_data(i,1)); % bus_i
    if Bus_data(i,2) == 0            % type
        Bus(i,2) = 3;
    elseif Bus_data(i,2) == 1
        Bus(i,2) = 2;
    else
        Bus(i,2) = 1;
    end
    Bus(i,3) = Bus_data(i,7);        % Pd
    Bus(i,4) = Bus_data(i,8);        % Qd
    % Bus(i,5) = 0                   % Gs
    % Bus(i,6) = 0                   % Bs
    Bus(i,7) = 1;                    % Area
    Bus(i,8) = 1;                    % Vm
    % Bus(i,9) = 0;                  % Va
    Bus(i,10) = Bus_data(1,12);      % Base kV 
    Bus(i,11) = 1;                   % zone
    Bus(i,12) = Bus_data(i, 10);     % Vmax
    Bus(i,13) = Bus_data(i,9);       % Vmin
end
mpc.bus = Bus;

%% generator data
%	bus	Pg	Qg	Qmax	Qmin	Vg	mBase	status	Pmax	Pmin	Pc1	Pc2	Qc1min	Qc1max	Qc2min	Qc2max	ramp_agc	ramp_10	ramp_30	ramp_q	apf
numgen = sum(Bus_data(:,2) == 1); % Hvor mange av bussene er generatorer?
numgen = numgen + 1;%Legger til slack-bus
Gen = zeros(numgen,21);%Oppretter en generator-matrise; Gen.
count = 1;
for i = 1:length(Bus_data(:,1))
   if Bus_data(i,2) == 0                            % Pmax
       Gen(count,8) = Bus_data(i,11)*10;    % For slack-bus setter vi Pmax til S_base*1000.
   elseif Bus_data(i,2) == 1
       Gen(count,8) = Bus_data(i,5);
   end
   if Bus_data(i,2) == 0 || Bus_data(i,2) == 1
      Gen(count,1) = Bus_data(i,1);                 % bus
      Gen(count,2) = Bus_data(i,5);                 % Pg
      Gen(count,3) = Bus_data(i,6);                 % Qg
      Gen(count,4) = 1000; %Bus_data(i,9);                 % Qmax
      Gen(count,5) = -1000; %Bus_data(i,10);                % Qmin
      Gen(count,6) = Bus_data(i,3)/Bus_data(1,12);  % Vg - Setter Vg(spenningen over gen.) til [pu]
      Gen(count,7) = Bus_data(1,11);                % mBase     
      Gen(count,8) = 1;                             % status
      count = count + 1;
   end
   
end
mpc.gen = Gen;

%% branch data
Branch = zeros(length(Branch_data(:,1)),13);
for i = 1:length(Branch_data(:,1))
    Branch(i,1) = round(Branch_data(i,1));  %fbus
    Branch(i,2) = round(Branch_data(i,2));  %tbus
    Branch(i,3) = Branch_data(i,3);         %r
    Branch(i,4) = Branch_data(i,4);         %x
    Branch(i,5) = Branch_data(i,5);         %b
    Branch(i,6) = 1000;                     %rateA
    Branch(i,7) = 1000;                     %rateB
    Branch(i,8) = 1000;                     %rateC
    % ratio = 0
    % angle = 0
    Branch(i,11) = 1;                       %status
    Branch(i,12) = -360;                    %angmin
    Branch(i,13) = 360;                     %angmin
end
mpc.branch = Branch;
