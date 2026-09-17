% ex2_plot.m
%
% Purpose:
%   Plot Example 2 error, solve time, and build-time comparisons.
%
% Problem:
%   Aging Maxwell second-kind Volterra equation solved by the VoltMat and
%   Gutleb--Olver scripts.
%
% Input files:
%   data/ex2_VoltMat_error.csv
%   data/ex2_VoltMat_time.csv
%   data/ex2_GO_error.csv
%   data/ex2_GO_time.csv
%
% Output files:
%   figures/ex2_error.pdf
%   figures/ex2_error.png
%   figures/ex2_time.pdf
%   figures/ex2_time.png
%   figures/ex2_build_time.pdf
%   figures/ex2_build_time.png
%
% Run:
%   matlab -batch "run('examples/ex2_2ndVolterraEqn/ex2_plot.m')"

clear;
clc;
close all;

% ============================================================
% Paths
% Put this script in examples/ex2_2ndVolterraEqn
% ============================================================

this_dir = fileparts(mfilename('fullpath'));
addpath(fullfile(fileparts(fileparts(this_dir)), 'tools'));
paperStyle = voltopt_plot_style();

data_dir = fullfile(this_dir, 'data');
fig_dir  = fullfile(this_dir, 'figures');

if ~exist(fig_dir, 'dir')
    mkdir(fig_dir);
end

csv_volt_error = fullfile(data_dir, 'ex2_VoltMat_error.csv');
csv_go_error   = fullfile(data_dir, 'ex2_GO_error.csv');

csv_volt_time = fullfile(data_dir, 'ex2_VoltMat_time.csv');
csv_go_time   = fullfile(data_dir, 'ex2_GO_time.csv');

% ============================================================
% Read CSV files
% ============================================================

T_volt_error = readtable(csv_volt_error);
T_go_error   = readtable(csv_go_error);

T_volt_time = readtable(csv_volt_time);
T_go_time   = readtable(csv_go_time);

% Keep successful rows if status column exists
T_volt_error = keep_ok_rows(T_volt_error);
T_go_error   = keep_ok_rows(T_go_error);

T_volt_time = keep_ok_rows(T_volt_time);
T_go_time   = keep_ok_rows(T_go_time);

% Sort by N
T_volt_error = sortrows(T_volt_error, 'N');
T_go_error   = sortrows(T_go_error, 'N');

T_volt_time = sortrows(T_volt_time, 'N');
T_go_time   = sortrows(T_go_time, 'N');

% ============================================================
% Column extraction
% ============================================================

N_volt_error = T_volt_error.N;
N_go_error   = T_go_error.N;

N_volt_time = T_volt_time.N;
N_go_time   = T_go_time.N;

err_volt = T_volt_error.relative_error;
err_go   = T_go_error.relative_error;

time_volt = get_column(T_volt_time, ...
    {'end2end_median_sec', 'end_to_end_median_sec'});
time_go = get_column(T_go_time, ...
    {'end2end_median_sec', 'end_to_end_median_sec'});

build_volt = get_column(T_volt_time, ...
    {'build_median_sec', 'operator_build_median_sec'});
build_go = get_column(T_go_time, ...
    {'build_median_sec', 'operator_build_median_sec'});

% Remove nonpositive or nonfinite values before log plots
idx_volt_err = ...
    isfinite(N_volt_error) & ...
    isfinite(err_volt) & ...
    err_volt > 0;

idx_go_err = ...
    isfinite(N_go_error) & ...
    isfinite(err_go) & ...
    err_go > 0;

idx_volt_time = ...
    isfinite(N_volt_time) & ...
    isfinite(time_volt) & ...
    time_volt > 0;

idx_go_time = ...
    isfinite(N_go_time) & ...
    isfinite(time_go) & ...
    time_go > 0;

idx_volt_build = ...
    isfinite(N_volt_time) & ...
    isfinite(build_volt) & ...
    build_volt > 0;

idx_go_build = ...
    isfinite(N_go_time) & ...
    isfinite(build_go) & ...
    build_go > 0;

% ============================================================
% Plot 1: error vs N
% Error CSV uses linearly spaced N
% ============================================================

fig1 = figure('Color', 'w', 'Units', 'centimeters', ...
    'Position', [2, 2, paperStyle.FigureSize]);

semilogy( ...
    N_volt_error(idx_volt_err), ...
    err_volt(idx_volt_err), ...
    '-', ...
    'LineWidth', paperStyle.PlotLineWidth);

hold on;

semilogy( ...
    N_go_error(idx_go_err), ...
    err_go(idx_go_err), ...
    '-', ...
    'LineWidth', paperStyle.PlotLineWidth);

xlabel('$N$', ...
    'Interpreter', 'latex');

ylabel('error', ...
    'Interpreter', 'latex');

legend( ...
    {'Duffy--Chebyshev', 'Gutleb--Olver'}, ...
    'Interpreter', 'latex', ...
    'Location', 'southwest');

ax = gca;

set(ax, ...
    'FontName', 'Times New Roman', ...
    'FontSize', 12);

box on;

% Keep MATLAB's native exponent label and its default relative position.
ax.XAxis.Exponent = 4;
drawnow;
voltopt_export_figure(fig1, ax, ...
    fullfile(fig_dir, 'ex2_error.pdf'), ...
    fullfile(fig_dir, 'ex2_error.png'));

% ============================================================
% Plot 2: end-to-end median time vs N
% Time CSV uses logarithmically spaced N
% ============================================================

fig2 = figure('Color', 'w', 'Units', 'centimeters', ...
    'Position', [2, 2, paperStyle.FigureSize]);

loglog( ...
    N_volt_time(idx_volt_time), ...
    time_volt(idx_volt_time), ...
    '-', ...
    'LineWidth', paperStyle.PlotLineWidth);

hold on;

loglog( ...
    N_go_time(idx_go_time), ...
    time_go(idx_go_time), ...
    '-', ...
    'LineWidth', paperStyle.PlotLineWidth);

xlabel('$N$', ...
    'Interpreter', 'latex');

ylabel('time (sec)', ...
    'Interpreter', 'latex');

legend( ...
    {'Duffy--Chebyshev', 'Gutleb--Olver'}, ...
    'Interpreter', 'latex', ...
    'Location', 'northwest');

axis([1e2, 3e4, 1e-3, 1e2]);

ax2 = gca;
set(ax2, ...
    'FontName', 'Times New Roman', ...
    'FontSize', 12);

box on;

voltopt_export_figure(fig2, ax2, ...
    fullfile(fig_dir, 'ex2_time.pdf'), ...
    fullfile(fig_dir, 'ex2_time.png'));

% ============================================================
% Plot 3: operator build median time vs N
% Time CSV uses logarithmically spaced N
% ============================================================

fig3 = figure('Color', 'w', 'Units', 'centimeters', ...
    'Position', [2, 2, paperStyle.FigureSize]);

loglog( ...
    N_volt_time(idx_volt_build), ...
    build_volt(idx_volt_build), ...
    '-', ...
    'LineWidth', paperStyle.PlotLineWidth);

hold on;

loglog( ...
    N_go_time(idx_go_build), ...
    build_go(idx_go_build), ...
    '-', ...
    'LineWidth', paperStyle.PlotLineWidth);

xlabel('$N$', ...
    'Interpreter', 'latex');

ylabel('build time (sec)', ...
    'Interpreter', 'latex');

legend( ...
    {'Duffy--Chebyshev', 'Gutleb--Olver'}, ...
    'Interpreter', 'latex', ...
    'Location', 'northwest');

axis([1e2, 3e4, 1e-3, 1e2]);

ax3 = gca;
set(ax3, ...
    'FontName', 'Times New Roman', ...
    'FontSize', 12);

box on;

voltopt_export_figure(fig3, ax3, ...
    fullfile(fig_dir, 'ex2_build_time.pdf'), ...
    fullfile(fig_dir, 'ex2_build_time.png'));

fprintf('Figures saved to:\n%s\n', fig_dir);

% ============================================================
% Helper functions
% ============================================================

function T = keep_ok_rows(T)

    if ismember('status', T.Properties.VariableNames)
        T = T(strcmp(string(T.status), "ok"), :);
    end

end

function x = get_column(T, names)

    vars = T.Properties.VariableNames;

    for k = 1:numel(names)

        name = names{k};

        if ismember(name, vars)
            x = T.(name);
            return;
        end

    end

    error( ...
        'None of the requested columns exist: %s', ...
        strjoin(names, ', '));

end
