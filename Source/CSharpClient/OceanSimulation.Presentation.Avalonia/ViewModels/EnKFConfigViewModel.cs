using CommunityToolkit.Mvvm.ComponentModel;
using OceanSimulation.Infrastructure.Interop;

namespace OceanSimulation.Presentation.Avalonia.ViewModels;

public partial class EnKFConfigViewModel : ViewModelBase
{
    [ObservableProperty]
    private int ensembleSize;

    [ObservableProperty]
    private double localizationRadius;

    [ObservableProperty]
    private double inflationFactor;

    [ObservableProperty]
    private double regularizationThreshold;

    [ObservableProperty]
    private bool useLocalization;

    [ObservableProperty]
    private bool useInflation;

    [ObservableProperty]
    private int numThreads;

    [ObservableProperty]
    private bool enableVectorization;

    public EnKFConfigViewModel()
    {
        var def = EnKFConfig.DefaultTOPAZ;
        ensembleSize = def.EnsembleSize;
        localizationRadius = def.LocalizationRadius;
        inflationFactor = def.InflationFactor;
        regularizationThreshold = def.RegularizationThreshold;
        useLocalization = def.UseLocalization;
        useInflation = def.UseInflation;
        numThreads = def.NumThreads;
        enableVectorization = def.EnableVectorization;
    }

    public EnKFConfig ToConfig() => new EnKFConfig
    {
        EnsembleSize = EnsembleSize,
        LocalizationRadius = LocalizationRadius,
        InflationFactor = InflationFactor,
        RegularizationThreshold = RegularizationThreshold,
        UseLocalization = UseLocalization,
        UseInflation = UseInflation,
        NumThreads = NumThreads,
        EnableVectorization = EnableVectorization
    };
}
