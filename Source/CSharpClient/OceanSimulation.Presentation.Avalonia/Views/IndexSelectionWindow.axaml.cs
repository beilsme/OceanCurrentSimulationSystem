using Avalonia.Controls;
using Avalonia.Interactivity;
using OceanSimulation.Presentation.Avalonia.ViewModels;

namespace OceanSimulation.Presentation.Avalonia.Views;

public partial class IndexSelectionWindow : Window
{
    public IndexSelectionWindow()
    {
        InitializeComponent();
        DataContext ??= new IndexSelectionViewModel();
    }

    public int TimeIndex => DataContext is IndexSelectionViewModel vm ? vm.TimeIndex : 0;
    public int DepthIndex => DataContext is IndexSelectionViewModel vm ? vm.DepthIndex : 0;

    private void Ok_Click(object? sender, RoutedEventArgs e)
    {
        Close(true);
    }
}
