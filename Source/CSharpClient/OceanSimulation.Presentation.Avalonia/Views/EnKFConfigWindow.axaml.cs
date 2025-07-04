using Avalonia.Controls;
using Avalonia.Interactivity;
using OceanSimulation.Infrastructure.Interop;
using OceanSimulation.Presentation.Avalonia.ViewModels;

namespace OceanSimulation.Presentation.Avalonia.Views;

public partial class EnKFConfigWindow : Window
{
    public EnKFConfigWindow()
    {
        InitializeComponent();
        DataContext ??= new EnKFConfigViewModel();
    }

    public EnKFConfig Config => DataContext is EnKFConfigViewModel vm ? vm.ToConfig() : EnKFConfig.DefaultTOPAZ;

    private void Ok_Click(object? sender, RoutedEventArgs e)
    {
        Close(true);
    }
}
