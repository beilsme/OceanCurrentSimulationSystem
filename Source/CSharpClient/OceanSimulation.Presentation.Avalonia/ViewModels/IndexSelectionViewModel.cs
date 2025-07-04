using CommunityToolkit.Mvvm.ComponentModel;

namespace OceanSimulation.Presentation.Avalonia.ViewModels;

public partial class IndexSelectionViewModel : ViewModelBase
{
    [ObservableProperty]
    private int timeIndex;

    [ObservableProperty]
    private int depthIndex;
}
