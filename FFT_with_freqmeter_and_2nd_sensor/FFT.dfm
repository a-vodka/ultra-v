object Form1: TForm1
  Left = 251
  Top = 140
  Caption = #1059#1083#1100#1090#1088#1072'-'#1042'-I'
  ClientHeight = 525
  ClientWidth = 984
  Color = clBtnFace
  Font.Charset = DEFAULT_CHARSET
  Font.Color = clWindowText
  Font.Height = -10
  Font.Name = 'MS Sans Serif'
  Font.Style = []
  OldCreateOrder = False
  OnShow = FormShow
  PixelsPerInch = 96
  TextHeight = 13
  object Panel1: TPanel
    Left = 765
    Top = 0
    Width = 219
    Height = 525
    Align = alRight
    TabOrder = 0
    object Label1: TLabel
      Left = 167
      Top = 473
      Width = 39
      Height = 13
      Caption = #1054#1073'/'#1084#1080#1085
    end
    object Label2: TLabel
      Left = 167
      Top = 495
      Width = 12
      Height = 13
      Caption = #1043#1094
    end
    object RadioGroup1: TRadioGroup
      Left = 3
      Top = 464
      Width = 94
      Height = 58
      Align = alCustom
      Caption = #1056#1077#1078#1080#1084
      ItemIndex = 0
      Items.Strings = (
        #1059#1089#1082#1086#1088#1077#1085#1080#1103
        #1050#1086#1076' '#1040#1062#1055
        #1042#1086#1083#1100#1090#1099)
      TabOrder = 0
      OnClick = RadioGroup1Click
    end
    object StringGrid1: TStringGrid
      Left = 1
      Top = 1
      Width = 217
      Height = 242
      Align = alTop
      ColCount = 3
      RowCount = 9
      Options = [goFixedVertLine, goFixedHorzLine, goVertLine, goHorzLine, goRangeSelect, goEditing]
      TabOrder = 1
      RowHeights = (
        24
        24
        24
        24
        24
        24
        24
        24
        24)
    end
    object Button2: TButton
      Left = 3
      Top = 302
      Width = 203
      Height = 54
      Caption = #1057#1090#1072#1088#1090
      TabOrder = 2
      OnClick = Button2Click
    end
    object Button3: TButton
      Left = 5
      Top = 391
      Width = 203
      Height = 53
      Caption = #1053#1072#1095#1072#1090#1100' '#1079#1072#1087#1080#1089#1100' '#1074' '#1092#1072#1081#1083
      TabOrder = 3
      OnClick = Button3Click
    end
    object Button1: TButton
      Left = 173
      Top = 362
      Width = 35
      Height = 25
      Caption = #1054#1090#1082#1088
      TabOrder = 4
      OnClick = Button1Click
    end
    object Edit3: TEdit
      Left = 4
      Top = 364
      Width = 162
      Height = 21
      TabOrder = 5
    end
    object CheckBox1: TCheckBox
      Left = 6
      Top = 252
      Width = 203
      Height = 17
      Caption = #1048#1089#1093#1086#1076#1085#1099#1081'/'#1054#1073#1088#1072#1073#1086#1090#1072#1085#1085#1099#1081' '#1089#1080#1075#1085#1072#1083
      Checked = True
      State = cbChecked
      TabOrder = 6
      OnClick = CheckBox1Click
    end
    object CheckBox2: TCheckBox
      Left = 6
      Top = 275
      Width = 203
      Height = 17
      Caption = #1064#1091#1084#1086#1087#1086#1076#1072#1074#1083#1077#1085#1080#1077
      Checked = True
      State = cbChecked
      TabOrder = 7
      OnClick = CheckBox2Click
    end
    object Edit1: TEdit
      Left = 103
      Top = 468
      Width = 58
      Height = 21
      TabOrder = 8
      Text = 'RPM'
    end
    object Edit2: TEdit
      Left = 103
      Top = 492
      Width = 58
      Height = 21
      TabOrder = 9
      Text = 'Hz'
    end
    object ProgressBar1: TProgressBar
      Left = 5
      Top = 445
      Width = 204
      Height = 17
      Max = 3000
      MarqueeInterval = 100
      TabOrder = 10
    end
  end
  object Panel2: TPanel
    Left = 0
    Top = 0
    Width = 765
    Height = 525
    Align = alClient
    TabOrder = 1
    object Splitter1: TSplitter
      Left = 1
      Top = 231
      Width = 763
      Height = 3
      Cursor = crVSplit
      Align = alTop
      ExplicitWidth = 293
    end
    object GroupBox2: TGroupBox
      Left = 1
      Top = 234
      Width = 763
      Height = 290
      Align = alClient
      Caption = #1057#1087#1077#1082#1090#1088
      TabOrder = 0
      object Chart2: TChart
        Left = 2
        Top = 15
        Width = 759
        Height = 273
        BackWall.Brush.Color = clWhite
        BackWall.Brush.Style = bsClear
        Legend.Visible = False
        Title.Text.Strings = (
          'TChart')
        Title.Visible = False
        DepthAxis.Automatic = False
        DepthAxis.AutomaticMaximum = False
        DepthAxis.AutomaticMinimum = False
        DepthAxis.Maximum = 0.650000000000001400
        DepthAxis.Minimum = -0.349999999999995000
        DepthTopAxis.Automatic = False
        DepthTopAxis.AutomaticMaximum = False
        DepthTopAxis.AutomaticMinimum = False
        DepthTopAxis.Maximum = 0.650000000000001400
        DepthTopAxis.Minimum = -0.349999999999995000
        RightAxis.Automatic = False
        RightAxis.AutomaticMaximum = False
        RightAxis.AutomaticMinimum = False
        View3D = False
        Align = alClient
        PopupMenu = PopupMenu1
        TabOrder = 0
        PrintMargins = (
          15
          39
          15
          39)
        ColorPaletteIndex = 13
        object Series3: TBarSeries
          BarPen.Visible = False
          Marks.Arrow.Visible = True
          Marks.Callout.Brush.Color = clBlack
          Marks.Callout.Arrow.Visible = True
          Marks.Callout.Length = 8
          Marks.Emboss.Color = 8487297
          Marks.Shadow.Color = 8487297
          Marks.Visible = False
          SeriesColor = clRed
          Transparency = 50
          BarWidthPercent = 100
          Dark3D = False
          Emboss.Color = 8947848
          MultiBar = mbNone
          Shadow.Color = 8947848
          Shadow.Visible = False
          SideMargins = False
          XValues.Name = 'X'
          XValues.Order = loAscending
          YValues.Name = 'Bar'
          YValues.Order = loNone
        end
        object Series4: TBarSeries
          BarPen.Visible = False
          Marks.Arrow.Visible = True
          Marks.Callout.Brush.Color = clBlack
          Marks.Callout.Arrow.Visible = True
          Marks.Callout.Length = 8
          Marks.Visible = False
          SeriesColor = clGreen
          Transparency = 50
          BarWidthPercent = 100
          Dark3D = False
          Emboss.Color = 8947848
          MultiBar = mbNone
          Shadow.Color = 8947848
          Shadow.Visible = False
          SideMargins = False
          XValues.Name = 'X'
          XValues.Order = loAscending
          YValues.Name = 'Bar'
          YValues.Order = loNone
        end
        object Series5: TPointSeries
          Marks.Arrow.Visible = True
          Marks.Callout.Brush.Color = clBlack
          Marks.Callout.Arrow.Visible = True
          Marks.Visible = False
          SeriesColor = clRed
          ClickableLine = False
          Pointer.Brush.Gradient.EndColor = clRed
          Pointer.Gradient.EndColor = clRed
          Pointer.InflateMargins = True
          Pointer.Style = psRectangle
          Pointer.Visible = True
          XValues.Name = 'X'
          XValues.Order = loAscending
          YValues.Name = 'Y'
          YValues.Order = loNone
        end
        object Series6: TPointSeries
          Marks.Arrow.Visible = True
          Marks.Callout.Brush.Color = clBlack
          Marks.Callout.Arrow.Visible = True
          Marks.Visible = False
          SeriesColor = clGreen
          ClickableLine = False
          Pointer.Brush.Gradient.EndColor = clGreen
          Pointer.Gradient.EndColor = clGreen
          Pointer.InflateMargins = True
          Pointer.Style = psRectangle
          Pointer.Visible = True
          XValues.Name = 'X'
          XValues.Order = loAscending
          YValues.Name = 'Y'
          YValues.Order = loNone
        end
        object Series9: TBarSeries
          Active = False
          BarBrush.Gradient.EndColor = 10708548
          BarPen.Visible = False
          Marks.Arrow.Visible = False
          Marks.Callout.Brush.Color = clBlack
          Marks.Callout.Arrow.Visible = False
          Marks.Visible = False
          SeriesColor = 16744448
          BarWidthPercent = 99
          Emboss.Color = 8684676
          Gradient.EndColor = 10708548
          MultiBar = mbNone
          Shadow.Color = 8684676
          Shadow.Smooth = False
          Shadow.Visible = False
          SideMargins = False
          XValues.Name = 'X'
          XValues.Order = loAscending
          YValues.Name = 'Bar'
          YValues.Order = loNone
        end
        object Series10: TBarSeries
          Active = False
          BarPen.Visible = False
          Marks.Arrow.Visible = True
          Marks.Callout.Brush.Color = clBlack
          Marks.Callout.Arrow.Visible = True
          Marks.Visible = False
          SeriesColor = 33023
          Transparency = 32
          BarWidthPercent = 99
          Emboss.Color = 8553090
          MultiBar = mbNone
          Shadow.Color = 8553090
          Shadow.Visible = False
          SideMargins = False
          XValues.Name = 'X'
          XValues.Order = loAscending
          YValues.Name = 'Bar'
          YValues.Order = loNone
        end
        object D_Spectrum_series: TBarSeries
          BarPen.Visible = False
          Marks.Arrow.Visible = True
          Marks.Callout.Brush.Color = clBlack
          Marks.Callout.Arrow.Visible = True
          Marks.Visible = False
          SeriesColor = clBlue
          Transparency = 50
          BarWidthPercent = 100
          Emboss.Color = 8684676
          MultiBar = mbNone
          Shadow.Color = 8684676
          SideMargins = False
          XValues.Name = 'X'
          XValues.Order = loAscending
          YValues.Name = 'Bar'
          YValues.Order = loNone
        end
        object E_Spectrum_series: TBarSeries
          BarPen.Visible = False
          Marks.Arrow.Visible = True
          Marks.Callout.Brush.Color = clBlack
          Marks.Callout.Arrow.Visible = True
          Marks.Visible = False
          SeriesColor = 33023
          Transparency = 53
          BarWidthPercent = 100
          Emboss.Color = 8816262
          MultiBar = mbNone
          Shadow.Color = 8816262
          SideMargins = False
          XValues.Name = 'X'
          XValues.Order = loAscending
          YValues.Name = 'Bar'
          YValues.Order = loNone
        end
        object F_Spectrum_series: TBarSeries
          BarPen.Visible = False
          Marks.Arrow.Visible = True
          Marks.Callout.Brush.Color = clBlack
          Marks.Callout.Arrow.Visible = True
          Marks.Visible = False
          SeriesColor = clPurple
          Transparency = 50
          BarWidthPercent = 100
          Emboss.Color = 8618883
          MultiBar = mbNone
          Shadow.Color = 8618883
          SideMargins = False
          XValues.Name = 'X'
          XValues.Order = loAscending
          YValues.Name = 'Bar'
          YValues.Order = loNone
        end
      end
    end
    object GroupBox1: TGroupBox
      Left = 1
      Top = 1
      Width = 763
      Height = 230
      Align = alTop
      Caption = #1057#1080#1075#1085#1072#1083
      TabOrder = 1
      object Chart1: TChart
        Left = 2
        Top = 49
        Width = 759
        Height = 179
        BackWall.Brush.Color = clWhite
        BackWall.Brush.Style = bsClear
        Legend.Alignment = laTop
        Legend.Visible = False
        Title.AdjustFrame = False
        Title.Alignment = taLeftJustify
        Title.Text.Strings = (
          'TChart')
        Title.Visible = False
        View3D = False
        Align = alClient
        TabOrder = 0
        PrintMargins = (
          15
          40
          15
          40)
        ColorPaletteIndex = 13
        object Series1: TFastLineSeries
          Marks.Arrow.Visible = True
          Marks.Callout.Brush.Color = clBlack
          Marks.Callout.Arrow.Visible = True
          Marks.Visible = False
          SeriesColor = clRed
          LinePen.Color = clRed
          TreatNulls = tnDontPaint
          XValues.Name = 'X'
          XValues.Order = loAscending
          YValues.Name = 'Y'
          YValues.Order = loNone
        end
        object Series2: TFastLineSeries
          Marks.Arrow.Visible = True
          Marks.Callout.Brush.Color = clBlack
          Marks.Callout.Arrow.Visible = True
          Marks.Visible = False
          SeriesColor = clGreen
          LinePen.Color = clGreen
          TreatNulls = tnDontPaint
          XValues.Name = 'X'
          XValues.Order = loAscending
          YValues.Name = 'Y'
          YValues.Order = loNone
        end
        object Series7: TFastLineSeries
          Active = False
          Marks.Arrow.Visible = True
          Marks.Callout.Brush.Color = clBlack
          Marks.Callout.Arrow.Visible = True
          Marks.Visible = False
          SeriesColor = clRed
          LinePen.Color = clRed
          TreatNulls = tnDontPaint
          XValues.Name = 'X'
          XValues.Order = loAscending
          YValues.Name = 'Y'
          YValues.Order = loNone
        end
        object Series8: TFastLineSeries
          Active = False
          Marks.Arrow.Visible = True
          Marks.Callout.Brush.Color = clBlack
          Marks.Callout.Arrow.Visible = True
          Marks.Visible = False
          SeriesColor = clGreen
          LinePen.Color = clGreen
          TreatNulls = tnDontPaint
          XValues.Name = 'X'
          XValues.Order = loAscending
          YValues.Name = 'Y'
          YValues.Order = loNone
        end
        object D_Series: TFastLineSeries
          Marks.Arrow.Visible = True
          Marks.Callout.Brush.Color = clBlack
          Marks.Callout.Arrow.Visible = True
          Marks.Visible = False
          SeriesColor = clBlue
          Title = 'D_Series'
          LinePen.Color = clBlue
          TreatNulls = tnDontPaint
          XValues.Name = 'X'
          XValues.Order = loAscending
          YValues.Name = 'Y'
          YValues.Order = loNone
        end
        object E_Series: TFastLineSeries
          Marks.Arrow.Visible = True
          Marks.Callout.Brush.Color = clBlack
          Marks.Callout.Arrow.Visible = True
          Marks.Visible = False
          SeriesColor = 33023
          Title = 'E_Series'
          LinePen.Color = 33023
          XValues.Name = 'X'
          XValues.Order = loAscending
          YValues.Name = 'Y'
          YValues.Order = loNone
        end
        object F_Series: TFastLineSeries
          Marks.Arrow.Visible = True
          Marks.Callout.Brush.Color = clBlack
          Marks.Callout.Arrow.Visible = True
          Marks.Visible = False
          SeriesColor = clPurple
          Title = 'F_Series'
          LinePen.Color = clPurple
          XValues.Name = 'X'
          XValues.Order = loAscending
          YValues.Name = 'Y'
          YValues.Order = loNone
        end
      end
      object TrackBar1: TTrackBar
        Left = 2
        Top = 15
        Width = 759
        Height = 34
        Align = alTop
        Max = 21
        Min = 1
        Position = 10
        TabOrder = 1
      end
    end
  end
  object OpenDialog1: TOpenDialog
    Left = 904
    Top = 136
  end
  object PopupMenu1: TPopupMenu
    Left = 840
    Top = 136
    object N1: TMenuItem
      AutoCheck = True
      Caption = #1043#1094
      Checked = True
      RadioItem = True
      OnClick = N1Click
    end
    object N2: TMenuItem
      AutoCheck = True
      Caption = #1056#1072#1076'/'#1089
      RadioItem = True
      OnClick = N2Click
    end
    object N3: TMenuItem
      AutoCheck = True
      Caption = #1054#1073'/'#1084#1080#1085
      RadioItem = True
      OnClick = N3Click
    end
  end
  object XPManifest1: TXPManifest
    Left = 840
    Top = 88
  end
end
