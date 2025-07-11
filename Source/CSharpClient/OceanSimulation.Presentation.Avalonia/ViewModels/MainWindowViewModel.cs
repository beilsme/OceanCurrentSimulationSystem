﻿using System;
using System.Collections.Generic;
using System.IO;
using System.Threading.Tasks;
using Avalonia.Media.Imaging;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Logging.Abstractions;
using OceanSimulation.Infrastructure.ComputeEngines;
using OceanSimulation.Infrastructure.Interop;

namespace OceanSimulation.Presentation.Avalonia.ViewModels;

public partial class MainWindowViewModel : ViewModelBase
{
    [ObservableProperty]
    private string? netcdfPath;

    [ObservableProperty]
    private Bitmap? resultImage;

    [ObservableProperty]
    private string status = "Ready";

    private readonly ILoggerFactory _loggerFactory;
    private OceanDataInterface? _dataInterface;
    private OceanStatisticalAnalysis? _analysis;
    private NetCDFParticleInterface? _particleInterface;
    private OceanAnimationInterface? _animationInterface;
    private PollutionDispersionInterface? _pollutionInterface;

    public MainWindowViewModel()
    {
        _loggerFactory = LoggerFactory.Create(builder => builder.AddConsole());
        VisualizeCommand = new AsyncRelayCommand(GenerateVisualizationAsync);
        AnimationCommand = new AsyncRelayCommand(GenerateAnimationAsync);
        VorticityCommand = new AsyncRelayCommand(GenerateVorticityAsync);
        ParticleCommand = new AsyncRelayCommand(RunParticleAsync);
        PollutionCommand = new AsyncRelayCommand(RunPollutionAsync);
        EnkfCommand = new AsyncRelayCommand(RunEnkfAsync);
    }

    public IAsyncRelayCommand VisualizeCommand { get; }
    public IAsyncRelayCommand AnimationCommand { get; }
    public IAsyncRelayCommand VorticityCommand { get; }
    public IAsyncRelayCommand ParticleCommand { get; }
    public IAsyncRelayCommand PollutionCommand { get; }
    public IAsyncRelayCommand EnkfCommand { get; }

    private async Task EnsureInitializedAsync()
    {
        if (_dataInterface != null)
            return;

        var config = new Dictionary<string, object>
        {
            ["PythonExecutablePath"] = "python3",
            ["PythonEngineRootPath"] = "../../../PythonEngine"
        };

        _dataInterface = new OceanDataInterface(_loggerFactory.CreateLogger<OceanDataInterface>(), config);
        await _dataInterface.InitializeAsync();
        _analysis = new OceanStatisticalAnalysis(_loggerFactory.CreateLogger<OceanStatisticalAnalysis>(), config);
        await _analysis.InitializeAsync();
        _particleInterface = new NetCDFParticleInterface(_loggerFactory.CreateLogger<NetCDFParticleInterface>(), config);
        await _particleInterface.InitializeAsync();
        _animationInterface = new OceanAnimationInterface(_loggerFactory.CreateLogger<OceanAnimationInterface>(), config);
        await _animationInterface.InitializeAsync();
        _pollutionInterface = new PollutionDispersionInterface(_loggerFactory.CreateLogger<PollutionDispersionInterface>(), config);
        await _pollutionInterface.InitializeAsync();
    }

    public async Task GenerateVisualizationAsync(int timeIndex, int depthIndex)
    {
        if (string.IsNullOrEmpty(NetcdfPath)) return;
        await EnsureInitializedAsync();
        Status = "Generating visualization...";
        var path = await _dataInterface!.GenerateVisualizationFromFileAsync(NetcdfPath, timeIndex, depthIndex);
        if (!string.IsNullOrEmpty(path) && File.Exists(path))
        {
            ResultImage = new Bitmap(path);
            Status = "Visualization complete";
        }
        else
        {
            Status = "Visualization failed";
        }
    }

    private Task GenerateVisualizationAsync()
    {
        return GenerateVisualizationAsync(0, 0);
    }


    public Task RunEnkfAsync(EnKFConfig config)
    {
        Status = $"EnKF parameters received: ensemble={config.EnsembleSize}";
        return Task.CompletedTask;
    }


    public async Task GenerateVorticityAsync(int timeIndex, int depthIndex)
    {
        if (string.IsNullOrEmpty(NetcdfPath)) return;
        await EnsureInitializedAsync();
        Status = "Calculating vorticity/divergence...";
        var path = await _analysis!.CalculateVorticityDivergenceFieldAsync(NetcdfPath, "", timeIndex, depthIndex);
        if (!string.IsNullOrEmpty(path) && File.Exists(path))
        {
            ResultImage = new Bitmap(path);
            Status = "Vorticity/Divergence complete";
        }
        else
        {
            Status = "Calculation failed";
        }
    }


    private Task GenerateVorticityAsync()
    {
        return GenerateVorticityAsync(0, 0);
    }

    private async Task RunParticleAsync()
    {
        if (string.IsNullOrEmpty(NetcdfPath)) return;
        await EnsureInitializedAsync();
        Status = "Running particle tracking...";
        var cfg = new ParticleTrackingConfig();
        var startLat = 24.5;
        var startLon = 120.0;
        var result = await _particleInterface!.TrackSingleParticleAsync(
            NetcdfPath, (startLat, startLon), cfg);
        if (result?.Success == true)
        {
            var viz = await _particleInterface.CreateTrajectoryVisualizationAsync(NetcdfPath, new[] { result.Trajectory });            if (viz?.Success == true && File.Exists(viz.OutputPath))
            {
                ResultImage = new Bitmap(viz.OutputPath);
                Status = "Particle tracking complete";
                return;
            }
        }
        Status = "Particle tracking failed";
    }

    private async Task GenerateAnimationAsync()
    {
        if (string.IsNullOrEmpty(NetcdfPath)) return;
        await EnsureInitializedAsync();
        Status = "Generating animation...";
        var path = await _animationInterface!.GenerateOceanAnimationAsync(NetcdfPath);
        if (!string.IsNullOrEmpty(path) && File.Exists(path))
        {
            ResultImage = new Bitmap(path);
            Status = "Animation complete";
        }
        else
        {
            Status = "Animation failed";
        }
    }

    private async Task RunPollutionAsync()
    {
        await EnsureInitializedAsync();
        Status = "Running pollution diffusion...";
        var result = await _pollutionInterface!.RunSimpleSimulationAsync();
        if (result?.Success == true && File.Exists(result.OutputPath))
        {
            ResultImage = new Bitmap(result.OutputPath);
            Status = "Pollution diffusion complete";
        }
        else
        {
            Status = "Pollution diffusion failed";
        }
    }

    private Task RunEnkfAsync()
    {
        Status = "EnKF prediction not implemented";
        return Task.CompletedTask;
    }
}
