clear; clc; close all;

csvfile = "ex5_volterra_schur_besselj1_pseudospectrum.csv";
T = readtable(csvfile);

x = unique(T.x, "sorted");
y = unique(T.y, "sorted");

nx = numel(x);
ny = numel(y);

% 假设 CSV 中顺序为固定 x 后遍历 y
Z = reshape(T.log10_pse, [ny, nx]).';

[X, Y] = meshgrid(x, y);
Zplot = Z.';

% 等值线层级：10^{-14}, ..., 10^{-1}
levels = -14:1:-1;

figure("Color", "w");

% 只画等高线，不填充，不显示数字
contour(X, Y, Zplot, levels, ...
    "LineWidth", 1.5);

% 最小值蓝色，最大值红色
colormap(turbo);
% 如果你的 Matlab 没有 turbo，就改成：
% colormap(jet);

cb = colorbar;
cb.Label.Interpreter = "latex";
cb.Label.String = "$\log_{10}(\epsilon)$";

% 保持 colorbar 数值方向正常：小值在下，大值在上
clim([min(levels), max(levels)]);
% 旧版本 Matlab 可改为：
% caxis([min(levels), max(levels)]);

xlabel("$\mathrm{Re}(z)$", "Interpreter", "latex");
ylabel("$\mathrm{Im}(z)$", "Interpreter", "latex");
title("Pseudospectrum", "Interpreter", "latex");

axis equal tight;
box on;
set(gca, "FontSize", 12, "LineWidth", 1.0);

% 输出 PDF
outpdf = "ex5_pseudospectrum.pdf";
exportgraphics(gcf, outpdf, ...
    "ContentType", "vector", ...
    "BackgroundColor", "white");