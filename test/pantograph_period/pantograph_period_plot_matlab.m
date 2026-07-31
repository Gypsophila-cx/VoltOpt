% Plot precomputed Julia data for the Iserles Figure 2 equation.
% MATLAB reads the curve only; it does not solve the differential equation.

% =========================================================
% User parameters
% =========================================================

script_fullpath = mfilename('fullpath');
script_dir = fileparts(script_fullpath);
data_file = fullfile( ...
    script_dir, 'data', 'pantograph_period_T1000_curve.csv');

rotation_angle_deg = 10.0;    % Counterclockwise rotation about the origin.
line_width = 0.5;
line_color = [0, 0, 0];
line_alpha = 1.0;             % See compatibility note below.
line_style = '-';

figure_width_px = 1000;
figure_height_px = 1000;
dpi = 300;

x_limits = [];               % Example: [-20, 20]; [] uses automatic limits.
y_limits = [];               % Example: [-20, 20]; [] uses automatic limits.
margin_fraction = 0.015;

save_png = true;
save_pdf = true;
display_result = true;
append_rotation_to_filename = false;

output_basename = fullfile( ...
    script_dir, 'figures', 'pantograph_period_T1000_matlab');

% =========================================================
% End of user parameters; no edits are normally needed below
% =========================================================

if ~isfile(data_file)
    error('Curve data not found: %s', data_file);
end

curve = readtable(data_file);
required_columns = {'t', 're_y', 'im_y'};
if ~all(ismember(required_columns, curve.Properties.VariableNames))
    error('CSV must contain columns: t, re_y, im_y');
end

x = curve.re_y;
y = curve.im_y;
if any(~isfinite(x)) || any(~isfinite(y))
    error('Curve data contain NaN or Inf.');
end

theta = deg2rad(rotation_angle_deg);
x_rot = cos(theta) .* x - sin(theta) .* y;
y_rot = sin(theta) .* x + cos(theta) .* y;

if display_result
    figure_visibility = 'on';
else
    figure_visibility = 'off';
end

fig = figure( ...
    'Color', 'w', ...
    'Units', 'pixels', ...
    'Position', [100, 100, figure_width_px, figure_height_px], ...
    'Visible', figure_visibility, ...
    'InvertHardcopy', 'off');
ax = axes(fig, 'Position', [0, 0, 1, 1]);

plot( ...
    ax, ...
    x_rot, ...
    y_rot, ...
    'Color', line_color, ...
    'LineWidth', line_width, ...
    'LineStyle', line_style);

% MATLAB Line objects do not consistently support alpha across releases.
% Keep line_alpha as a user-facing parameter, but only opaque lines are used
% here to avoid version-dependent or invalid RGBA Color assignments.
if line_alpha ~= 1.0
    warning( ...
        'line_alpha is retained for compatibility; this MATLAB Line is opaque.');
end

axis(ax, 'equal');

if isempty(x_limits) && isempty(y_limits)
    x_mid = 0.5 * (min(x_rot) + max(x_rot));
    y_mid = 0.5 * (min(y_rot) + max(y_rot));
    square_span = max(max(x_rot) - min(x_rot), max(y_rot) - min(y_rot));
    square_span = max(square_span, eps);
    half_span = 0.5 * square_span * (1 + 2 * margin_fraction);
    xlim(ax, x_mid + [-half_span, half_span]);
    ylim(ax, y_mid + [-half_span, half_span]);
else
    if ~isempty(x_limits)
        xlim(ax, x_limits);
    end
    if ~isempty(y_limits)
        ylim(ax, y_limits);
    end
end

axis(ax, 'off');
set( ...
    ax, ...
    'Position', [0, 0, 1, 1], ...
    'LooseInset', [0, 0, 0, 0], ...
    'Visible', 'off');
drawnow;

if append_rotation_to_filename && abs(rotation_angle_deg) > eps
    angle_text = strrep(sprintf('%g', rotation_angle_deg), '.', 'p');
    angle_text = strrep(angle_text, '-', 'm');
    angle_text = strrep(angle_text, '+', 'p');
    actual_basename = sprintf('%s_rot%s', output_basename, angle_text);
else
    actual_basename = output_basename;
end

output_directory = fileparts(actual_basename);
if ~isfolder(output_directory)
    mkdir(output_directory);
end

written_files = {};
if save_png
    png_path = [actual_basename, '.png'];
    exportgraphics( ...
        fig, ...
        png_path, ...
        'Resolution', dpi, ...
        'BackgroundColor', 'white');
    written_files{end + 1} = png_path; %#ok<SAGROW>
end
if save_pdf
    pdf_path = [actual_basename, '.pdf'];
    exportgraphics( ...
        fig, ...
        pdf_path, ...
        'ContentType', 'vector', ...
        'BackgroundColor', 'white');
    written_files{end + 1} = pdf_path; %#ok<SAGROW>
end

fprintf('MATLAB Figure 2 phase plot completed.\n');
fprintf('  data points: %d\n', height(curve));
fprintf('  rotation angle: %.12g degrees\n', rotation_angle_deg);
fprintf('  input: %s\n', data_file);
for index = 1:numel(written_files)
    fprintf('  output: %s\n', written_files{index});
end

if ~display_result
    close(fig);
end
