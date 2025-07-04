using Avalonia.Controls;
using Avalonia.Interactivity;
using System;
using System.Collections.Generic;
using System.Threading.Tasks;
using OceanSimulation.Presentation.Avalonia.ViewModels;

namespace OceanSimulation.Presentation.Avalonia.Views;

public partial class MainWindow : Window
{
    public MainWindow()
    {
        InitializeComponent();
    }

    private async void BrowseFile_Click(object? sender, RoutedEventArgs e)
    {
        var dialog = new OpenFileDialog
        {
            AllowMultiple = false,
            Directory = Environment.GetFolderPath(Environment.SpecialFolder.UserProfile),
            Filters = new List<FileDialogFilter>
            {
                new FileDialogFilter { Name = "NetCDF", Extensions = { "nc" } },
                new FileDialogFilter { Name = "All Files", Extensions = { "*" } }
            }
        };

        var result = await dialog.ShowAsync(this);
        if (result != null && result.Length > 0 && DataContext is MainWindowViewModel vm)
        {
            vm.NetcdfPath = result[0];
        }
    }
}
