<?xml version='1.0' encoding='utf-8'?>
<q:quakeml xmlns="http://quakeml.org/xmlns/bed/1.2" xmlns:q="http://quakeml.org/xmlns/quakeml/1.2">
  <eventParameters publicID="smi:org.gfz-potsdam.de/geofon/EventParameters">
    <event publicID="smi:org.gfz-potsdam.de/geofon/texnet2022wmmd">
      <preferredOriginID>smi:org.gfz-potsdam.de/geofon/NLL.20221117170218.53264.116304</preferredOriginID>
      <preferredMagnitudeID>smi:org.gfz-potsdam.de/geofon/Magnitude/20221117170222.739536.116338</preferredMagnitudeID>
      <preferredFocalMechanismID>smi:org.gfz-potsdam.de/geofon/FocalMechanism/20221117003721.714512.92750</preferredFocalMechanismID>
      <type>earthquake</type>
      <typeCertainty>known</typeCertainty>
      <description>
        <text>Western Texas</text>
        <type>region name</type>
      </description>
      <creationInfo>
        <agencyID>TXNet</agencyID>
        <author>scevent@sc3primary.beg.utexas.edu</author>
        <creationTime>2022-11-16T21:32:55.683979Z</creationTime>
      </creationInfo>
      <origin publicID="smi:org.gfz-potsdam.de/geofon/NLL.20221117170218.53264.116304">
        <time>
          <value>2022-11-16T21:32:44.481830Z</value>
        </time>
        <latitude>
          <value>31.6367452</value>
          <uncertainty>0.2193609021</uncertainty>
        </latitude>
        <longitude>
          <value>-103.9987871</value>
          <uncertainty>0.2936453823</uncertainty>
        </longitude>
        <depth>
          <value>6902.807617</value>
          <uncertainty>629.622792</uncertainty>
        </depth>
        <methodID>smi:org.gfz-potsdam.de/geofon/NonLinLoc</methodID>
        <earthModelID>smi:org.gfz-potsdam.de/geofon/PB1D-20170918-topoLV</earthModelID>
        <quality>
          <associatedPhaseCount>119</associatedPhaseCount>
          <usedPhaseCount>55</usedPhaseCount>
          <usedStationCount>42</usedStationCount>
          <standardError>0.09978100572</standardError>
          <azimuthalGap>46.07142564</azimuthalGap>
          <secondaryAzimuthalGap>62.72676132</secondaryAzimuthalGap>
          <groundTruthLevel>-</groundTruthLevel>
          <minimumDistance>0.02345152448</minimumDistance>
          <maximumDistance>0.3749718621</maximumDistance>
          <medianDistance>0.3150239362</medianDistance>
        </quality>
        <evaluationMode>manual</evaluationMode>
        <evaluationStatus>final</evaluationStatus>
        <creationInfo>
          <agencyID>TXNet</agencyID>
          <author>daniel</author>
          <creationTime>2022-11-17T17:02:19.722683Z</creationTime>
        </creationInfo>
        <originUncertainty>
          <horizontalUncertainty>462.8109061</horizontalUncertainty>
          <minHorizontalUncertainty>307.9021627</minHorizontalUncertainty>
          <maxHorizontalUncertainty>462.8109061</maxHorizontalUncertainty>
          <azimuthMaxHorizontalUncertainty>68.61737477</azimuthMaxHorizontalUncertainty>
          <confidenceEllipsoid>
            <semiMajorAxisLength>1247078.413</semiMajorAxisLength>
            <semiMinorAxisLength>377909.4905</semiMinorAxisLength>
            <semiIntermediateAxisLength>419051.1494</semiIntermediateAxisLength>
            <majorAxisPlunge>19.61495438</majorAxisPlunge>
            <majorAxisAzimuth>-161.4637085</majorAxisAzimuth>
            <majorAxisRotation>95.96235939</majorAxisRotation>
          </confidenceEllipsoid>
        </originUncertainty>
        <arrival publicID="smi:org.gfz-potsdam.de/geofon/Pick/20221116213904.882278.81130_NLL.20221117170218.53264.116304">
          <pickID>smi:org.gfz-potsdam.de/geofon/Pick/20221116213904.882278.81130</pickID>
          <phase>S</phase>
          <timeCorrection>0.0</timeCorrection>
          <azimuth>114.7184316</azimuth>
          <distance>0.4421683437</distance>
          <takeoffAngle>
            <value>91.9</value>
          </takeoffAngle>
          <timeResidual>0.0495881889</timeResidual>
          <timeWeight>0.7303527392</timeWeight>
        </arrival>
        <arrival publicID="smi:org.gfz-potsdam.de/geofon/Pick/20221116213904.88235.81131_NLL.20221117170218.53264.116304">
          <pickID>smi:org.gfz-potsdam.de/geofon/Pick/20221116213904.88235.81131</pickID>
          <phase>S</phase>
          <timeCorrection>0.0</timeCorrection>
          <azimuth>118.7803512</azimuth>
          <distance>0.4755707842</distance>
          <takeoffAngle>
            <value>91.8</value>
          </takeoffAngle>
          <timeResidual>-0.2109771772</timeResidual>
          <timeWeight>0.6242999354</timeWeight>
        </arrival>
        <arrival publicID="smi:org.gfz-potsdam.de/geofon/Pick/20221116214756.54407.84373_NLL.20221117170218.53264.116304">
          <pickID>smi:org.gfz-potsdam.de/geofon/Pick/20221116214756.54407.84373</pickID>
          <phase>P</phase>
          <timeCorrection>0.0</timeCorrection>
          <azimuth>320.5225295</azimuth>
          <distance>0.05202981106</distance>
          <takeoffAngle>
            <value>138.2</value>
          </takeoffAngle>
          <timeResidual>0.1767528044</timeResidual>
          <timeWeight>0.9235935826</timeWeight>
        </arrival>
        <arrival publicID="smi:org.gfz-potsdam.de/geofon/Pick/20221116214756.544252.84374_NLL.20221117170218.53264.116304">
          <pickID>smi:org.gfz-potsdam.de/geofon/Pick/20221116214756.544252.84374</pickID>
          <phase>S</phase>
          <timeCorrection>0.0</timeCorrection>
          <azimuth>320.5225295</azimuth>
          <distance>0.05202981106</distance>
          <takeoffAngle>
            <value>138.2</value>
          </takeoffAngle>
          <timeResidual>-0.01797428259</timeResidual>
          <timeWeight>1.001004023</timeWeight>
        </arrival>
        <arrival publicID="smi:org.gfz-potsdam.de/geofon/Pick/20221116214756.544409.84377_NLL.20221117170218.53264.116304">
          <pickID>smi:org.gfz-potsdam.de/geofon/Pick/20221116214756.544409.84377</pickID>
          <phase>P</phase>
          <timeCorrection>0.0</timeCorrection>
          <azimuth>328.6132401</azimuth>
          <distance>0.09994640016</distance>
          <takeoffAngle>
            <value>112.8</value>
          </takeoffAngle>
          <timeResidual>0.1543081093</timeResidual>
          <timeWeight>0.8761622957</timeWeight>
        </arrival>
        <arrival publicID="smi:org.gfz-potsdam.de/geofon/Pick/20221116214756.544453.84378_NLL.20221117170218.53264.116304">
          <pickID>smi:org.gfz-potsdam.de/geofon/Pick/20221116214756.544453.84378</pickID>
          <phase>S</phase>
          <timeCorrection>0.0</timeCorrection>
          <azimuth>328.6132401</azimuth>
          <distance>0.09994640016</distance>
          <takeoffAngle>
            <value>112.8</value>
          </takeoffAngle>
          <timeResidual>0.02776428281</timeResidual>
          <timeWeight>1.002165365</timeWeight>
        </arrival>
        <arrival publicID="smi:org.gfz-potsdam.de/geofon/Pick/20221116214756.544502.84379_NLL.20221117170218.53264.116304">
          <pickID>smi:org.gfz-potsdam.de/geofon/Pick/20221116214756.544502.84379</pickID>
          <phase>S</phase>
          <timeCorrection>0.0</timeCorrection>
          <azimuth>313.3900991</azimuth>
          <distance>0.1762233021</distance>
          <takeoffAngle>
            <value>97.6</value>
          </takeoffAngle>
          <timeResidual>-0.6361252515</timeResidual>
          <timeWeight>0.0</timeWeight>
        </arrival>
        <arrival publicID="smi:org.gfz-potsdam.de/geofon/Pick/20221116214756.544546.84380_NLL.20221117170218.53264.116304">
          <pickID>smi:org.gfz-potsdam.de/geofon/Pick/20221116214756.544546.84380</pickID>
          <phase>P</phase>
          <timeCorrection>0.0</timeCorrection>
          <azimuth>313.3900991</azimuth>
          <distance>0.1762233021</distance>
          <takeoffAngle>
            <value>97.6</value>
          </takeoffAngle>
          <timeResidual>0.0699025821</timeResidual>
          <timeWeight>0.9799628791</timeWeight>
        </arrival>
        <arrival publicID="smi:org.gfz-potsdam.de/geofon/Pick/20221116214756.544593.84381_NLL.20221117170218.53264.116304">
          <pickID>smi:org.gfz-potsdam.de/geofon/Pick/20221116214756.544593.84381</pickID>
          <phase>P</phase>
          <timeCorrection>0.0</timeCorrection>
          <azimuth>50.50793085</azimuth>
          <distance>0.1836456002</distance>
          <takeoffAngle>
            <value>97.0</value>
          </takeoffAngle>
          <timeResidual>0.169116343</timeResidual>
          <timeWeight>0.8513160003</timeWeight>
        </arrival>
        <arrival publicID="smi:org.gfz-potsdam.de/geofon/Pick/20221116214756.544638.84382_NLL.20221117170218.53264.116304">
          <pickID>smi:org.gfz-potsdam.de/geofon/Pick/20221116214756.544638.84382</pickID>
          <phase>S</phase>
          <timeCorrection>0.0</timeCorrection>
          <azimuth>50.50793085</azimuth>
          <distance>0.1836456002</distance>
          <takeoffAngle>
            <value>97.0</value>
          </takeoffAngle>
          <timeResidual>0.1732865526</timeResidual>
          <timeWeight>0.7093860465</timeWeight>
        </arrival>
        <arrival publicID="smi:org.gfz-potsdam.de/geofon/Pick/20221116214756.544684.84383_NLL.20221117170218.53264.116304">
          <pickID>smi:org.gfz-potsdam.de/geofon/Pick/20221116214756.544684.84383</pickID>
          <phase>P</phase>
          <timeCorrection>0.0</timeCorrection>
          <azimuth>279.0456291</azimuth>
          <distance>0.2295749498</distance>
          <takeoffAngle>
            <value>94.8</value>
          </takeoffAngle>
          <timeResidual>0.08240049835</timeResidual>
          <timeWeight>0.7767698517</timeWeight>
        </arrival>
        <arrival publicID="smi:org.gfz-potsdam.de/geofon/Pick/20221116215249.515307.86784_NLL.20221117170218.53264.116304">
          <pickID>smi:org.gfz-potsdam.de/geofon/Pick/20221116215249.515307.86784</pickID>
          <phase>S</phase>
          <timeCorrection>0.0</timeCorrection>
          <azimuth>279.0456291</azimuth>
          <distance>0.2295749498</distance>
          <takeoffAngle>
            <value>94.8</value>
          </takeoffAngle>
          <timeResidual>-0.7410304915</timeResidual>
          <timeWeight>0.0</timeWeight>
        </arrival>
        <arrival publicID="smi:org.gfz-potsdam.de/geofon/Pick/20221116215249.51548.86785_NLL.20221117170218.53264.116304">
          <pickID>smi:org.gfz-potsdam.de/geofon/Pick/20221116215249.51548.86785</pickID>
          <phase>S</phase>
          <timeCorrection>0.0</timeCorrection>
          <azimuth>298.9960161</azimuth>
          <distance>0.2485396667</distance>
          <takeoffAngle>
            <value>94.3</value>
          </takeoffAngle>
          <timeResidual>-0.6181800209</timeResidual>
          <timeWeight>0.0</timeWeight>
        </arrival>
        <arrival publicID="smi:org.gfz-potsdam.de/geofon/Pick/20221116215249.515542.86786_NLL.20221117170218.53264.116304">
          <pickID>smi:org.gfz-potsdam.de/geofon/Pick/20221116215249.515542.86786</pickID>
          <phase>P</phase>
          <timeCorrection>0.0</timeCorrection>
          <azimuth>298.9960161</azimuth>
          <distance>0.2485396667</distance>
          <takeoffAngle>
            <value>94.3</value>
          </takeoffAngle>
          <timeResidual>-0.06822705621</timeResidual>
          <timeWeight>1.190912669</timeWeight>
        </arrival>
        <arrival publicID="smi:org.gfz-potsdam.de/geofon/Pick/20221116215249.515612.86787_NLL.20221117170218.53264.116304">
          <pickID>smi:org.gfz-potsdam.de/geofon/Pick/20221116215249.515612.86787</pickID>
          <phase>S</phase>
          <timeCorrection>0.0</timeCorrection>
          <azimuth>157.3088203</azimuth>
          <distance>0.2894799386</distance>
          <takeoffAngle>
            <value>93.4</value>
          </takeoffAngle>
          <timeResidual>-1.001496503</timeResidual>
          <timeWeight>0.0</timeWeight>
        </arrival>
        <arrival publicID="smi:org.gfz-potsdam.de/geofon/Pick/20221116215249.515669.86788_NLL.20221117170218.53264.116304">
          <pickID>smi:org.gfz-potsdam.de/geofon/Pick/20221116215249.515669.86788</pickID>
          <phase>P</phase>
          <timeCorrection>0.0</timeCorrection>
          <azimuth>157.3088203</azimuth>
          <distance>0.2894799386</distance>
          <takeoffAngle>
            <value>93.4</value>
          </takeoffAngle>
          <timeResidual>-0.3936884181</timeResidual>
          <timeWeight>0.0</timeWeight>
        </arrival>
        <arrival publicID="smi:org.gfz-potsdam.de/geofon/Pick/20221116215249.515835.86791_NLL.20221117170218.53264.116304">
          <pickID>smi:org.gfz-potsdam.de/geofon/Pick/20221116215249.515835.86791</pickID>
          <phase>P</phase>
          <timeCorrection>0.0</timeCorrection>
          <azimuth>149.6311203</azimuth>
          <distance>0.4106894349</distance>
          <takeoffAngle>
            <value>92.1</value>
          </takeoffAngle>
          <timeResidual>-0.3054726966</timeResidual>
          <timeWeight>0.0</timeWeight>
        </arrival>
        <arrival publicID="smi:org.gfz-potsdam.de/geofon/Pick/20221116215249.515889.86792_NLL.20221117170218.53264.116304">
          <pickID>smi:org.gfz-potsdam.de/geofon/Pick/20221116215249.515889.86792</pickID>
          <phase>S</phase>
          <timeCorrection>0.0</timeCorrection>
          <azimuth>149.6311203</azimuth>
          <distance>0.4106894349</distance>
          <takeoffAngle>
            <value>92.1</value>
          </takeoffAngle>
          <timeResidual>-0.6322191986</timeResidual>
          <timeWeight>0.0</timeWeight>
        </arrival>
        <arrival publicID="smi:org.gfz-potsdam.de/geofon/Pick/20221116215249.515945.86793_NLL.20221117170218.53264.116304">
          <pickID>smi:org.gfz-potsdam.de/geofon/Pick/20221116215249.515945.86793</pickID>
          <phase>P</phase>
          <timeCorrection>0.0</timeCorrection>
          <azimuth>175.484673</azimuth>
          <distance>0.4256101256</distance>
          <takeoffAngle>
            <value>92.0</value>
          </takeoffAngle>
          <timeResidual>-0.4059632931</timeResidual>
          <timeWeight>0.0</timeWeight>
        </arrival>
        <arrival publicID="smi:org.gfz-potsdam.de/geofon/Pick/20221116215249.516053.86795_NLL.20221117170218.53264.116304">
          <pickID>smi:org.gfz-potsdam.de/geofon/Pick/20221116215249.516053.86795</pickID>
          <phase>P</phase>
          <timeCorrection>0.0</timeCorrection>
          <azimuth>274.6230519</azimuth>
          <distance>0.4292169728</distance>
          <takeoffAngle>
            <value>92.0</value>
          </takeoffAngle>
          <timeResidual>-0.2739291123</timeResidual>
          <timeWeight>0.0</timeWeight>
        </arrival>
        <arrival publicID="smi:org.gfz-potsdam.de/geofon/Pick/20221116215249.516168.86797_NLL.20221117170218.53264.116304">
          <pickID>smi:org.gfz-potsdam.de/geofon/Pick/20221116215249.516168.86797</pickID>
          <phase>P</phase>
          <timeCorrection>0.0</timeCorrection>
          <azimuth>142.324561</azimuth>
          <distance>0.5141647447</distance>
          <takeoffAngle>
            <value>91.7</value>
          </takeoffAngle>
          <timeResidual>-0.2205056102</timeResidual>
          <timeWeight>0.7127222459</timeWeight>
        </arrival>
        <arrival publicID="smi:org.gfz-potsdam.de/geofon/Pick/20221116215249.516222.86798_NLL.20221117170218.53264.116304">
          <pickID>smi:org.gfz-potsdam.de/geofon/Pick/20221116215249.516222.86798</pickID>
          <phase>P</phase>
          <timeCorrection>0.0</timeCorrection>
          <azimuth>89.12392521</azimuth>
          <distance>0.6657049965</distance>
          <takeoffAngle>
            <value>89.8</value>
          </takeoffAngle>
          <timeResidual>-0.1868519873</timeResidual>
          <timeWeight>0.8255633232</timeWeight>
        </arrival>
        <arrival publicID="smi:org.gfz-potsdam.de/geofon/Pick/20221116215249.516274.86799_NLL.20221117170218.53264.116304">
          <pickID>smi:org.gfz-potsdam.de/geofon/Pick/20221116215249.516274.86799</pickID>
          <phase>S</phase>
          <timeCorrection>0.0</timeCorrection>
          <azimuth>89.12392521</azimuth>
          <distance>0.6657049965</distance>
          <takeoffAngle>
            <value>89.8</value>
          </takeoffAngle>
          <timeResidual>-1.535557459</timeResidual>
          <timeWeight>0.0</timeWeight>
        </arrival>
        <arrival publicID="smi:org.gfz-potsdam.de/geofon/Pick/20221116215249.516327.86800_NLL.20221117170218.53264.116304">
          <pickID>smi:org.gfz-potsdam.de/geofon/Pick/20221116215249.516327.86800</pickID>
          <phase>P</phase>
          <timeCorrection>0.0</timeCorrection>
          <azimuth>121.7639879</azimuth>
          <distance>0.6790601459</distance>
          <takeoffAngle>
            <value>87.8</value>
          </takeoffAngle>
          <timeResidual>-0.03542309012</timeResidual>
          <timeWeight>1.091569313</timeWeight>
        </arrival>
        <arrival publicID="smi:org.gfz-potsdam.de/geofon/Pick/20221116215249.516374.86801_NLL.20221117170218.53264.116304">
          <pickID>smi:org.gfz-potsdam.de/geofon/Pick/20221116215249.516374.86801</pickID>
          <phase>S</phase>
          <timeCorrection>0.0</timeCorrection>
          <azimuth>121.7639879</azimuth>
          <distance>0.6790601459</distance>
          <takeoffAngle>
            <value>87.9</value>
          </takeoffAngle>
          <timeResidual>-0.9554084788</timeResidual>
          <timeWeight>0.0</timeWeight>
        </arrival>
        <arrival publicID="smi:org.gfz-potsdam.de/geofon/Pick/20221116215249.516427.86802_NLL.20221117170218.53264.116304">
          <pickID>smi:org.gfz-potsdam.de/geofon/Pick/20221116215249.516427.86802</pickID>
          <phase>P</phase>
          <timeCorrection>0.0</timeCorrection>
          <azimuth>143.2789943</azimuth>
          <distance>0.6919191011</distance>
          <takeoffAngle>
            <value>85.3</value>
          </takeoffAngle>
          <timeResidual>-0.3707963675</timeResidual>
          <timeWeight>0.0</timeWeight>
        </arrival>
        <arrival publicID="smi:org.gfz-potsdam.de/geofon/Pick/20221116215249.516479.86803_NLL.20221117170218.53264.116304">
          <pickID>smi:org.gfz-potsdam.de/geofon/Pick/20221116215249.516479.86803</pickID>
          <phase>S</phase>
          <timeCorrection>0.0</timeCorrection>
          <azimuth>143.2789943</azimuth>
          <distance>0.6919191011</distance>
          <takeoffAngle>
            <value>85.5</value>
          </takeoffAngle>
          <timeResidual>-1.291722763</timeResidual>
          <timeWeight>0.0</timeWeight>
        </arrival>
        <arrival publicID="smi:org.gfz-potsdam.de/geofon/Pick/20221116215503.577442.88046_NLL.20221117170218.53264.116304">
          <pickID>smi:org.gfz-potsdam.de/geofon/Pick/20221116215503.577442.88046</pickID>
          <phase>P</phase>
          <timeCorrection>0.0</timeCorrection>
          <azimuth>114.7184316</azimuth>
          <distance>0.4421683437</distance>
          <takeoffAngle>
            <value>91.9</value>
          </takeoffAngle>
          <timeResidual>0.2013714857</timeResidual>
          <timeWeight>0.7792584495</timeWeight>
        </arrival>
        <arrival publicID="smi:org.gfz-potsdam.de/geofon/Pick/20221116215503.577625.88047_NLL.20221117170218.53264.116304">
          <pickID>smi:org.gfz-potsdam.de/geofon/Pick/20221116215503.577625.88047</pickID>
          <phase>P</phase>
          <timeCorrection>0.0</timeCorrection>
          <azimuth>118.7803512</azimuth>
          <distance>0.4755707842</distance>
          <takeoffAngle>
            <value>91.8</value>
          </takeoffAngle>
          <timeResidual>0.0771454498</timeResidual>
          <timeWeight>0.9379914005</timeWeight>
        </arrival>
        <arrival publicID="smi:org.gfz-potsdam.de/geofon/Pick/20221116215503.577692.88048_NLL.20221117170218.53264.116304">
          <pickID>smi:org.gfz-potsdam.de/geofon/Pick/20221116215503.577692.88048</pickID>
          <phase>P</phase>
          <timeCorrection>0.0</timeCorrection>
          <azimuth>136.2029496</azimuth>
          <distance>0.5101243886</distance>
          <takeoffAngle>
            <value>91.7</value>
          </takeoffAngle>
          <timeResidual>-0.1756757063</timeResidual>
          <timeWeight>0.7916265968</timeWeight>
        </arrival>
        <arrival publicID="smi:org.gfz-potsdam.de/geofon/Pick/20221117015938.385144.74510_NLL.20221117170218.53264.116304">
          <pickID>smi:org.gfz-potsdam.de/geofon/Pick/20221117015938.385144.74510</pickID>
          <phase>P</phase>
          <timeCorrection>0.0</timeCorrection>
          <azimuth>223.8660953</azimuth>
          <distance>0.02345152448</distance>
          <takeoffAngle>
            <value>158.1</value>
          </takeoffAngle>
          <timeResidual>-0.03648824508</timeResidual>
          <timeWeight>1.245844371</timeWeight>
        </arrival>
        <arrival publicID="smi:org.gfz-potsdam.de/geofon/Pick/20221117015938.385263.74511_NLL.20221117170218.53264.116304">
          <pickID>smi:org.gfz-potsdam.de/geofon/Pick/20221117015938.385263.74511</pickID>
          <phase>P</phase>
          <timeCorrection>0.0</timeCorrection>
          <azimuth>135.827024</azimuth>
          <distance>0.03651137465</distance>
          <takeoffAngle>
            <value>148.4</value>
          </takeoffAngle>
          <timeResidual>-0.04800092438</timeResidual>
          <timeWeight>1.230866736</timeWeight>
        </arrival>
        <arrival publicID="smi:org.gfz-potsdam.de/geofon/Pick/20221117015938.3853.74512_NLL.20221117170218.53264.116304">
          <pickID>smi:org.gfz-potsdam.de/geofon/Pick/20221117015938.3853.74512</pickID>
          <phase>P</phase>
          <timeCorrection>0.0</timeCorrection>
          <azimuth>261.0899022</azimuth>
          <distance>0.05442637709</distance>
          <takeoffAngle>
            <value>136.8</value>
          </takeoffAngle>
          <timeResidual>0.05683002911</timeResidual>
          <timeWeight>1.232799673</timeWeight>
        </arrival>
        <arrival publicID="smi:org.gfz-potsdam.de/geofon/Pick/20221117015938.385627.74518_NLL.20221117170218.53264.116304">
          <pickID>smi:org.gfz-potsdam.de/geofon/Pick/20221117015938.385627.74518</pickID>
          <phase>S</phase>
          <timeCorrection>0.0</timeCorrection>
          <azimuth>293.6520466</azimuth>
          <distance>0.06873102299</distance>
          <takeoffAngle>
            <value>126.7</value>
          </takeoffAngle>
          <timeResidual>-0.07912815783</timeResidual>
          <timeWeight>1.173799933</timeWeight>
        </arrival>
        <arrival publicID="smi:org.gfz-potsdam.de/geofon/Pick/20221117015938.385718.74520_NLL.20221117170218.53264.116304">
          <pickID>smi:org.gfz-potsdam.de/geofon/Pick/20221117015938.385718.74520</pickID>
          <phase>P</phase>
          <timeCorrection>0.0</timeCorrection>
          <azimuth>228.8291483</azimuth>
          <distance>0.08122707231</distance>
          <takeoffAngle>
            <value>120.4</value>
          </takeoffAngle>
          <timeResidual>0.001926146611</timeResidual>
          <timeWeight>1.269744759</timeWeight>
        </arrival>
        <arrival publicID="smi:org.gfz-potsdam.de/geofon/Pick/20221117015938.385851.74521_NLL.20221117170218.53264.116304">
          <pickID>smi:org.gfz-potsdam.de/geofon/Pick/20221117015938.385851.74521</pickID>
          <phase>P</phase>
          <timeCorrection>0.0</timeCorrection>
          <azimuth>344.4484955</azimuth>
          <distance>0.09946870646</distance>
          <takeoffAngle>
            <value>112.8</value>
          </takeoffAngle>
          <timeResidual>0.1557468939</timeResidual>
          <timeWeight>0.9925355677</timeWeight>
        </arrival>
        <arrival publicID="smi:org.gfz-potsdam.de/geofon/Pick/20221117015938.385932.74523_NLL.20221117170218.53264.116304">
          <pickID>smi:org.gfz-potsdam.de/geofon/Pick/20221117015938.385932.74523</pickID>
          <phase>P</phase>
          <timeCorrection>0.0</timeCorrection>
          <azimuth>158.9798967</azimuth>
          <distance>0.114177377</distance>
          <takeoffAngle>
            <value>108.3</value>
          </takeoffAngle>
          <timeResidual>-0.06737632851</timeResidual>
          <timeWeight>1.198055333</timeWeight>
        </arrival>
        <arrival publicID="smi:org.gfz-potsdam.de/geofon/Pick/20221117015938.386056.74526_NLL.20221117170218.53264.116304">
          <pickID>smi:org.gfz-potsdam.de/geofon/Pick/20221117015938.386056.74526</pickID>
          <phase>P</phase>
          <timeCorrection>0.0</timeCorrection>
          <azimuth>24.93790857</azimuth>
          <distance>0.1097855479</distance>
          <takeoffAngle>
            <value>109.5</value>
          </takeoffAngle>
          <timeResidual>0.1370960715</timeResidual>
          <timeWeight>1.050212931</timeWeight>
        </arrival>
        <arrival publicID="smi:org.gfz-potsdam.de/geofon/Pick/20221117015938.38619.74529_NLL.20221117170218.53264.116304">
          <pickID>smi:org.gfz-potsdam.de/geofon/Pick/20221117015938.38619.74529</pickID>
          <phase>P</phase>
          <timeCorrection>0.0</timeCorrection>
          <azimuth>122.5413655</azimuth>
          <distance>0.1545129781</distance>
          <takeoffAngle>
            <value>99.8</value>
          </takeoffAngle>
          <timeResidual>0.02710291317</timeResidual>
          <timeWeight>1.263020769</timeWeight>
        </arrival>
        <arrival publicID="smi:org.gfz-potsdam.de/geofon/Pick/20221117015938.386355.74530_NLL.20221117170218.53264.116304">
          <pickID>smi:org.gfz-potsdam.de/geofon/Pick/20221117015938.386355.74530</pickID>
          <phase>P</phase>
          <timeCorrection>0.0</timeCorrection>
          <azimuth>340.5216767</azimuth>
          <distance>0.2273640798</distance>
          <takeoffAngle>
            <value>94.9</value>
          </takeoffAngle>
          <timeResidual>-0.07042877044</timeResidual>
          <timeWeight>1.191162887</timeWeight>
        </arrival>
        <arrival publicID="smi:org.gfz-potsdam.de/geofon/Pick/20221117015938.386418.74531_NLL.20221117170218.53264.116304">
          <pickID>smi:org.gfz-potsdam.de/geofon/Pick/20221117015938.386418.74531</pickID>
          <phase>S</phase>
          <timeCorrection>0.0</timeCorrection>
          <azimuth>340.5216767</azimuth>
          <distance>0.2273640798</distance>
          <takeoffAngle>
            <value>94.9</value>
          </takeoffAngle>
          <timeResidual>-0.2775805784</timeResidual>
          <timeWeight>0.0</timeWeight>
        </arrival>
        <arrival publicID="smi:org.gfz-potsdam.de/geofon/Pick/20221117015938.38662.74533_NLL.20221117170218.53264.116304">
          <pickID>smi:org.gfz-potsdam.de/geofon/Pick/20221117015938.38662.74533</pickID>
          <phase>P</phase>
          <timeCorrection>0.0</timeCorrection>
          <azimuth>335.3040824</azimuth>
          <distance>0.2940766591</distance>
          <takeoffAngle>
            <value>93.3</value>
          </takeoffAngle>
          <timeResidual>-0.174538084</timeResidual>
          <timeWeight>0.0</timeWeight>
        </arrival>
        <arrival publicID="smi:org.gfz-potsdam.de/geofon/Pick/20221117015938.386661.74534_NLL.20221117170218.53264.116304">
          <pickID>smi:org.gfz-potsdam.de/geofon/Pick/20221117015938.386661.74534</pickID>
          <phase>S</phase>
          <timeCorrection>0.0</timeCorrection>
          <azimuth>335.3040824</azimuth>
          <distance>0.2940766591</distance>
          <takeoffAngle>
            <value>93.3</value>
          </takeoffAngle>
          <timeResidual>-0.341939286</timeResidual>
          <timeWeight>0.0</timeWeight>
        </arrival>
        <arrival publicID="smi:org.gfz-potsdam.de/geofon/Pick/20221117015938.386699.74535_NLL.20221117170218.53264.116304">
          <pickID>smi:org.gfz-potsdam.de/geofon/Pick/20221117015938.386699.74535</pickID>
          <phase>P</phase>
          <timeCorrection>0.0</timeCorrection>
          <azimuth>354.4987141</azimuth>
          <distance>0.30076649</distance>
          <takeoffAngle>
            <value>93.2</value>
          </takeoffAngle>
          <timeResidual>-0.1832168478</timeResidual>
          <timeWeight>0.0</timeWeight>
        </arrival>
        <arrival publicID="smi:org.gfz-potsdam.de/geofon/Pick/20221117015938.386736.74536_NLL.20221117170218.53264.116304">
          <pickID>smi:org.gfz-potsdam.de/geofon/Pick/20221117015938.386736.74536</pickID>
          <phase>S</phase>
          <timeCorrection>0.0</timeCorrection>
          <azimuth>354.4987141</azimuth>
          <distance>0.30076649</distance>
          <takeoffAngle>
            <value>93.2</value>
          </takeoffAngle>
          <timeResidual>0.006012002967</timeResidual>
          <timeWeight>0.9474917792</timeWeight>
        </arrival>
        <arrival publicID="smi:org.gfz-potsdam.de/geofon/Pick/20221117015938.386792.74537_NLL.20221117170218.53264.116304">
          <pickID>smi:org.gfz-potsdam.de/geofon/Pick/20221117015938.386792.74537</pickID>
          <phase>P</phase>
          <timeCorrection>0.0</timeCorrection>
          <azimuth>37.50801486</azimuth>
          <distance>0.3292813824</distance>
          <takeoffAngle>
            <value>92.9</value>
          </takeoffAngle>
          <timeResidual>-0.02073126258</timeResidual>
          <timeWeight>1.232055425</timeWeight>
        </arrival>
        <arrival publicID="smi:org.gfz-potsdam.de/geofon/Pick/20221117015938.386847.74538_NLL.20221117170218.53264.116304">
          <pickID>smi:org.gfz-potsdam.de/geofon/Pick/20221117015938.386847.74538</pickID>
          <phase>S</phase>
          <timeCorrection>0.0</timeCorrection>
          <azimuth>37.50801486</azimuth>
          <distance>0.3292813824</distance>
          <takeoffAngle>
            <value>92.9</value>
          </takeoffAngle>
          <timeResidual>0.730621682</timeResidual>
          <timeWeight>0.0</timeWeight>
        </arrival>
        <arrival publicID="smi:org.gfz-potsdam.de/geofon/Pick/20221117015938.386886.74539_NLL.20221117170218.53264.116304">
          <pickID>smi:org.gfz-potsdam.de/geofon/Pick/20221117015938.386886.74539</pickID>
          <phase>P</phase>
          <timeCorrection>0.0</timeCorrection>
          <azimuth>67.39702014</azimuth>
          <distance>0.3313377372</distance>
          <takeoffAngle>
            <value>92.8</value>
          </takeoffAngle>
          <timeResidual>0.1226399137</timeResidual>
          <timeWeight>1.076326191</timeWeight>
        </arrival>
        <arrival publicID="smi:org.gfz-potsdam.de/geofon/Pick/20221117015938.386917.74540_NLL.20221117170218.53264.116304">
          <pickID>smi:org.gfz-potsdam.de/geofon/Pick/20221117015938.386917.74540</pickID>
          <phase>P</phase>
          <timeCorrection>0.0</timeCorrection>
          <azimuth>342.082456</azimuth>
          <distance>0.337543089</distance>
          <takeoffAngle>
            <value>92.8</value>
          </takeoffAngle>
          <timeResidual>-0.2614763441</timeResidual>
          <timeWeight>0.0</timeWeight>
        </arrival>
        <arrival publicID="smi:org.gfz-potsdam.de/geofon/Pick/20221117015938.386951.74541_NLL.20221117170218.53264.116304">
          <pickID>smi:org.gfz-potsdam.de/geofon/Pick/20221117015938.386951.74541</pickID>
          <phase>S</phase>
          <timeCorrection>0.0</timeCorrection>
          <azimuth>342.082456</azimuth>
          <distance>0.337543089</distance>
          <takeoffAngle>
            <value>92.8</value>
          </takeoffAngle>
          <timeResidual>-0.3003198659</timeResidual>
          <timeWeight>0.0</timeWeight>
        </arrival>
        <arrival publicID="smi:org.gfz-potsdam.de/geofon/Pick/20221117015938.386988.74542_NLL.20221117170218.53264.116304">
          <pickID>smi:org.gfz-potsdam.de/geofon/Pick/20221117015938.386988.74542</pickID>
          <phase>P</phase>
          <timeCorrection>0.0</timeCorrection>
          <azimuth>30.27372744</azimuth>
          <distance>0.3368589945</distance>
          <takeoffAngle>
            <value>92.8</value>
          </takeoffAngle>
          <timeResidual>-0.01947494107</timeResidual>
          <timeWeight>1.230467495</timeWeight>
        </arrival>
        <arrival publicID="smi:org.gfz-potsdam.de/geofon/Pick/20221117015938.387074.74544_NLL.20221117170218.53264.116304">
          <pickID>smi:org.gfz-potsdam.de/geofon/Pick/20221117015938.387074.74544</pickID>
          <phase>P</phase>
          <timeCorrection>0.0</timeCorrection>
          <azimuth>273.3612114</azimuth>
          <distance>0.369954708</distance>
          <takeoffAngle>
            <value>92.5</value>
          </takeoffAngle>
          <timeResidual>-0.2706330539</timeResidual>
          <timeWeight>0.0</timeWeight>
        </arrival>
        <arrival publicID="smi:org.gfz-potsdam.de/geofon/Pick/20221117015938.387093.74545_NLL.20221117170218.53264.116304">
          <pickID>smi:org.gfz-potsdam.de/geofon/Pick/20221117015938.387093.74545</pickID>
          <phase>P</phase>
          <timeCorrection>0.0</timeCorrection>
          <azimuth>284.6630917</azimuth>
          <distance>0.3763898323</distance>
          <takeoffAngle>
            <value>92.4</value>
          </takeoffAngle>
          <timeResidual>-0.2599838735</timeResidual>
          <timeWeight>0.0</timeWeight>
        </arrival>
        <arrival publicID="smi:org.gfz-potsdam.de/geofon/Pick/20221117015938.387124.74546_NLL.20221117170218.53264.116304">
          <pickID>smi:org.gfz-potsdam.de/geofon/Pick/20221117015938.387124.74546</pickID>
          <phase>P</phase>
          <timeCorrection>0.0</timeCorrection>
          <azimuth>27.11868045</azimuth>
          <distance>0.3673737909</distance>
          <takeoffAngle>
            <value>92.5</value>
          </takeoffAngle>
          <timeResidual>0.01669843685</timeResidual>
          <timeWeight>1.226781608</timeWeight>
        </arrival>
        <arrival publicID="smi:org.gfz-potsdam.de/geofon/Pick/20221117015938.387156.74547_NLL.20221117170218.53264.116304">
          <pickID>smi:org.gfz-potsdam.de/geofon/Pick/20221117015938.387156.74547</pickID>
          <phase>P</phase>
          <timeCorrection>0.0</timeCorrection>
          <azimuth>8.991273174</azimuth>
          <distance>0.3749718621</distance>
          <takeoffAngle>
            <value>92.4</value>
          </takeoffAngle>
          <timeResidual>-0.05278373579</timeResidual>
          <timeWeight>1.184548475</timeWeight>
        </arrival>
        <arrival publicID="smi:org.gfz-potsdam.de/geofon/Pick/20221117015938.387195.74548_NLL.20221117170218.53264.116304">
          <pickID>smi:org.gfz-potsdam.de/geofon/Pick/20221117015938.387195.74548</pickID>
          <phase>P</phase>
          <timeCorrection>0.0</timeCorrection>
          <azimuth>306.6757199</azimuth>
          <distance>0.3870923724</distance>
          <takeoffAngle>
            <value>92.3</value>
          </takeoffAngle>
          <timeResidual>-0.3618665702</timeResidual>
          <timeWeight>0.0</timeWeight>
        </arrival>
        <arrival publicID="smi:org.gfz-potsdam.de/geofon/Pick/20221117015938.387232.74549_NLL.20221117170218.53264.116304">
          <pickID>smi:org.gfz-potsdam.de/geofon/Pick/20221117015938.387232.74549</pickID>
          <phase>P</phase>
          <timeCorrection>0.0</timeCorrection>
          <azimuth>38.87077318</azimuth>
          <distance>0.3769795666</distance>
          <takeoffAngle>
            <value>92.4</value>
          </takeoffAngle>
          <timeResidual>0.009112625407</timeResidual>
          <timeWeight>1.224985864</timeWeight>
        </arrival>
        <arrival publicID="smi:org.gfz-potsdam.de/geofon/Pick/20221117015938.387366.74550_NLL.20221117170218.53264.116304">
          <pickID>smi:org.gfz-potsdam.de/geofon/Pick/20221117015938.387366.74550</pickID>
          <phase>P</phase>
          <timeCorrection>0.0</timeCorrection>
          <azimuth>33.5558089</azimuth>
          <distance>0.4329661882</distance>
          <takeoffAngle>
            <value>92.0</value>
          </takeoffAngle>
          <timeResidual>-0.08797198086</timeResidual>
          <timeWeight>1.106863592</timeWeight>
        </arrival>
        <arrival publicID="smi:org.gfz-potsdam.de/geofon/Pick/20221117015938.387442.74551_NLL.20221117170218.53264.116304">
          <pickID>smi:org.gfz-potsdam.de/geofon/Pick/20221117015938.387442.74551</pickID>
          <phase>P</phase>
          <timeCorrection>0.0</timeCorrection>
          <azimuth>285.2039322</azimuth>
          <distance>0.4546590154</distance>
          <takeoffAngle>
            <value>91.9</value>
          </takeoffAngle>
          <timeResidual>-0.4699734103</timeResidual>
          <timeWeight>0.0</timeWeight>
        </arrival>
        <arrival publicID="smi:org.gfz-potsdam.de/geofon/Pick/20221117015938.387489.74552_NLL.20221117170218.53264.116304">
          <pickID>smi:org.gfz-potsdam.de/geofon/Pick/20221117015938.387489.74552</pickID>
          <phase>P</phase>
          <timeCorrection>0.0</timeCorrection>
          <azimuth>280.6478966</azimuth>
          <distance>0.4588507684</distance>
          <takeoffAngle>
            <value>91.9</value>
          </takeoffAngle>
          <timeResidual>-0.368789471</timeResidual>
          <timeWeight>0.0</timeWeight>
        </arrival>
        <arrival publicID="smi:org.gfz-potsdam.de/geofon/Pick/20221117015938.387526.74553_NLL.20221117170218.53264.116304">
          <pickID>smi:org.gfz-potsdam.de/geofon/Pick/20221117015938.387526.74553</pickID>
          <phase>P</phase>
          <timeCorrection>0.0</timeCorrection>
          <azimuth>121.3101983</azimuth>
          <distance>0.4548760947</distance>
          <takeoffAngle>
            <value>91.9</value>
          </takeoffAngle>
          <timeResidual>0.03041140548</timeResidual>
          <timeWeight>0.9659311943</timeWeight>
        </arrival>
        <arrival publicID="smi:org.gfz-potsdam.de/geofon/Pick/20221117015938.387596.74554_NLL.20221117170218.53264.116304">
          <pickID>smi:org.gfz-potsdam.de/geofon/Pick/20221117015938.387596.74554</pickID>
          <phase>P</phase>
          <timeCorrection>0.0</timeCorrection>
          <azimuth>323.4357129</azimuth>
          <distance>0.4546293402</distance>
          <takeoffAngle>
            <value>91.9</value>
          </takeoffAngle>
          <timeResidual>-0.5925287396</timeResidual>
          <timeWeight>0.0</timeWeight>
        </arrival>
        <arrival publicID="smi:org.gfz-potsdam.de/geofon/Pick/20221117015938.387635.74555_NLL.20221117170218.53264.116304">
          <pickID>smi:org.gfz-potsdam.de/geofon/Pick/20221117015938.387635.74555</pickID>
          <phase>P</phase>
          <timeCorrection>0.0</timeCorrection>
          <azimuth>29.50511714</azimuth>
          <distance>0.4586819082</distance>
          <takeoffAngle>
            <value>91.9</value>
          </takeoffAngle>
          <timeResidual>-0.08918370382</timeResidual>
          <timeWeight>1.097296229</timeWeight>
        </arrival>
        <arrival publicID="smi:org.gfz-potsdam.de/geofon/Pick/20221117015938.387729.74556_NLL.20221117170218.53264.116304">
          <pickID>smi:org.gfz-potsdam.de/geofon/Pick/20221117015938.387729.74556</pickID>
          <phase>P</phase>
          <timeCorrection>0.0</timeCorrection>
          <azimuth>14.65028508</azimuth>
          <distance>0.4682437261</distance>
          <takeoffAngle>
            <value>91.8</value>
          </takeoffAngle>
          <timeResidual>-0.125471782</timeResidual>
          <timeWeight>1.01292249</timeWeight>
        </arrival>
        <arrival publicID="smi:org.gfz-potsdam.de/geofon/Pick/20221117015938.387764.74557_NLL.20221117170218.53264.116304">
          <pickID>smi:org.gfz-potsdam.de/geofon/Pick/20221117015938.387764.74557</pickID>
          <phase>P</phase>
          <timeCorrection>0.0</timeCorrection>
          <azimuth>122.6568187</azimuth>
          <distance>0.49244757</distance>
          <takeoffAngle>
            <value>91.7</value>
          </takeoffAngle>
          <timeResidual>0.03098179533</timeResidual>
          <timeWeight>0.9584865258</timeWeight>
        </arrival>
        <arrival publicID="smi:org.gfz-potsdam.de/geofon/Pick/20221117015938.387801.74558_NLL.20221117170218.53264.116304">
          <pickID>smi:org.gfz-potsdam.de/geofon/Pick/20221117015938.387801.74558</pickID>
          <phase>P</phase>
          <timeCorrection>0.0</timeCorrection>
          <azimuth>117.1958352</azimuth>
          <distance>0.4932828739</distance>
          <takeoffAngle>
            <value>91.7</value>
          </takeoffAngle>
          <timeResidual>0.09308756722</timeResidual>
          <timeWeight>0.9215535744</timeWeight>
        </arrival>
        <arrival publicID="smi:org.gfz-potsdam.de/geofon/Pick/20221117015938.387838.74559_NLL.20221117170218.53264.116304">
          <pickID>smi:org.gfz-potsdam.de/geofon/Pick/20221117015938.387838.74559</pickID>
          <phase>P</phase>
          <timeCorrection>0.0</timeCorrection>
          <azimuth>152.4515032</azimuth>
          <distance>0.5048801691</distance>
          <takeoffAngle>
            <value>91.7</value>
          </takeoffAngle>
          <timeResidual>-0.4726831946</timeResidual>
          <timeWeight>0.0</timeWeight>
        </arrival>
        <arrival publicID="smi:org.gfz-potsdam.de/geofon/Pick/20221117015938.387875.74560_NLL.20221117170218.53264.116304">
          <pickID>smi:org.gfz-potsdam.de/geofon/Pick/20221117015938.387875.74560</pickID>
          <phase>P</phase>
          <timeCorrection>0.0</timeCorrection>
          <azimuth>4.788517721</azimuth>
          <distance>0.4931506059</distance>
          <takeoffAngle>
            <value>91.7</value>
          </takeoffAngle>
          <timeResidual>-0.2133103617</timeResidual>
          <timeWeight>0.7597295903</timeWeight>
        </arrival>
        <arrival publicID="smi:org.gfz-potsdam.de/geofon/Pick/20221117015938.387909.74561_NLL.20221117170218.53264.116304">
          <pickID>smi:org.gfz-potsdam.de/geofon/Pick/20221117015938.387909.74561</pickID>
          <phase>P</phase>
          <timeCorrection>0.0</timeCorrection>
          <azimuth>119.5868922</azimuth>
          <distance>0.5048911493</distance>
          <takeoffAngle>
            <value>91.7</value>
          </takeoffAngle>
          <timeResidual>0.0531248487</timeResidual>
          <timeWeight>0.9474914991</timeWeight>
        </arrival>
        <arrival publicID="smi:org.gfz-potsdam.de/geofon/Pick/20221117015938.388009.74562_NLL.20221117170218.53264.116304">
          <pickID>smi:org.gfz-potsdam.de/geofon/Pick/20221117015938.388009.74562</pickID>
          <phase>P</phase>
          <timeCorrection>0.0</timeCorrection>
          <azimuth>42.98725227</azimuth>
          <distance>0.5033884852</distance>
          <takeoffAngle>
            <value>91.7</value>
          </takeoffAngle>
          <timeResidual>-0.2422632456</timeResidual>
          <timeWeight>0.0</timeWeight>
        </arrival>
        <arrival publicID="smi:org.gfz-potsdam.de/geofon/Pick/20221117020158.518175.74663_NLL.20221117170218.53264.116304">
          <pickID>smi:org.gfz-potsdam.de/geofon/Pick/20221117020158.518175.74663</pickID>
          <phase>S</phase>
          <timeCorrection>0.0</timeCorrection>
          <azimuth>67.39702014</azimuth>
          <distance>0.3313377372</distance>
          <takeoffAngle>
            <value>92.8</value>
          </takeoffAngle>
          <timeResidual>0.1739906311</timeResidual>
          <timeWeight>0.8056784117</timeWeight>
        </arrival>
        <arrival publicID="smi:org.gfz-potsdam.de/geofon/Pick/20221117020158.518298.74664_NLL.20221117170218.53264.116304">
          <pickID>smi:org.gfz-potsdam.de/geofon/Pick/20221117020158.518298.74664</pickID>
          <phase>S</phase>
          <timeCorrection>0.0</timeCorrection>
          <azimuth>30.27372744</azimuth>
          <distance>0.3368589945</distance>
          <takeoffAngle>
            <value>92.8</value>
          </takeoffAngle>
          <timeResidual>1.760261162</timeResidual>
          <timeWeight>0.0</timeWeight>
        </arrival>
        <arrival publicID="smi:org.gfz-potsdam.de/geofon/Pick/20221117020158.51843.74666_NLL.20221117170218.53264.116304">
          <pickID>smi:org.gfz-potsdam.de/geofon/Pick/20221117020158.51843.74666</pickID>
          <phase>S</phase>
          <timeCorrection>0.0</timeCorrection>
          <azimuth>273.3612114</azimuth>
          <distance>0.369954708</distance>
          <takeoffAngle>
            <value>92.5</value>
          </takeoffAngle>
          <timeResidual>-1.652062349</timeResidual>
          <timeWeight>0.0</timeWeight>
        </arrival>
        <arrival publicID="smi:org.gfz-potsdam.de/geofon/Pick/20221117020158.51849.74667_NLL.20221117170218.53264.116304">
          <pickID>smi:org.gfz-potsdam.de/geofon/Pick/20221117020158.51849.74667</pickID>
          <phase>S</phase>
          <timeCorrection>0.0</timeCorrection>
          <azimuth>284.6630917</azimuth>
          <distance>0.3763898323</distance>
          <takeoffAngle>
            <value>92.4</value>
          </takeoffAngle>
          <timeResidual>-1.142357174</timeResidual>
          <timeWeight>0.0</timeWeight>
        </arrival>
        <arrival publicID="smi:org.gfz-potsdam.de/geofon/Pick/20221117020158.518547.74668_NLL.20221117170218.53264.116304">
          <pickID>smi:org.gfz-potsdam.de/geofon/Pick/20221117020158.518547.74668</pickID>
          <phase>S</phase>
          <timeCorrection>0.0</timeCorrection>
          <azimuth>27.11868045</azimuth>
          <distance>0.3673737909</distance>
          <takeoffAngle>
            <value>92.5</value>
          </takeoffAngle>
          <timeResidual>1.833968345</timeResidual>
          <timeWeight>0.0</timeWeight>
        </arrival>
        <arrival publicID="smi:org.gfz-potsdam.de/geofon/Pick/20221117020158.518622.74669_NLL.20221117170218.53264.116304">
          <pickID>smi:org.gfz-potsdam.de/geofon/Pick/20221117020158.518622.74669</pickID>
          <phase>S</phase>
          <timeCorrection>0.0</timeCorrection>
          <azimuth>8.991273174</azimuth>
          <distance>0.3749718621</distance>
          <takeoffAngle>
            <value>92.4</value>
          </takeoffAngle>
          <timeResidual>0.02874052917</timeResidual>
          <timeWeight>0.9158117675</timeWeight>
        </arrival>
        <arrival publicID="smi:org.gfz-potsdam.de/geofon/Pick/20221117020158.518706.74670_NLL.20221117170218.53264.116304">
          <pickID>smi:org.gfz-potsdam.de/geofon/Pick/20221117020158.518706.74670</pickID>
          <phase>S</phase>
          <timeCorrection>0.0</timeCorrection>
          <azimuth>306.6757199</azimuth>
          <distance>0.3870923724</distance>
          <takeoffAngle>
            <value>92.3</value>
          </takeoffAngle>
          <timeResidual>-1.28684367</timeResidual>
          <timeWeight>0.0</timeWeight>
        </arrival>
        <arrival publicID="smi:org.gfz-potsdam.de/geofon/Pick/20221117020158.518822.74671_NLL.20221117170218.53264.116304">
          <pickID>smi:org.gfz-potsdam.de/geofon/Pick/20221117020158.518822.74671</pickID>
          <phase>S</phase>
          <timeCorrection>0.0</timeCorrection>
          <azimuth>38.87077318</azimuth>
          <distance>0.3769795666</distance>
          <takeoffAngle>
            <value>92.4</value>
          </takeoffAngle>
          <timeResidual>0.8944543156</timeResidual>
          <timeWeight>0.0</timeWeight>
        </arrival>
        <arrival publicID="smi:org.gfz-potsdam.de/geofon/Pick/20221117020158.518935.74672_NLL.20221117170218.53264.116304">
          <pickID>smi:org.gfz-potsdam.de/geofon/Pick/20221117020158.518935.74672</pickID>
          <phase>S</phase>
          <timeCorrection>0.0</timeCorrection>
          <azimuth>175.484673</azimuth>
          <distance>0.4256101256</distance>
          <takeoffAngle>
            <value>92.0</value>
          </takeoffAngle>
          <timeResidual>-1.171546046</timeResidual>
          <timeWeight>0.0</timeWeight>
        </arrival>
        <arrival publicID="smi:org.gfz-potsdam.de/geofon/Pick/20221117020158.519003.74673_NLL.20221117170218.53264.116304">
          <pickID>smi:org.gfz-potsdam.de/geofon/Pick/20221117020158.519003.74673</pickID>
          <phase>S</phase>
          <timeCorrection>0.0</timeCorrection>
          <azimuth>274.6230519</azimuth>
          <distance>0.4292169728</distance>
          <takeoffAngle>
            <value>92.0</value>
          </takeoffAngle>
          <timeResidual>-1.074087214</timeResidual>
          <timeWeight>0.0</timeWeight>
        </arrival>
        <arrival publicID="smi:org.gfz-potsdam.de/geofon/Pick/20221117020158.51915.74674_NLL.20221117170218.53264.116304">
          <pickID>smi:org.gfz-potsdam.de/geofon/Pick/20221117020158.51915.74674</pickID>
          <phase>S</phase>
          <timeCorrection>0.0</timeCorrection>
          <azimuth>280.6478966</azimuth>
          <distance>0.4588507684</distance>
          <takeoffAngle>
            <value>91.9</value>
          </takeoffAngle>
          <timeResidual>-1.191958643</timeResidual>
          <timeWeight>0.0</timeWeight>
        </arrival>
        <arrival publicID="smi:org.gfz-potsdam.de/geofon/Pick/20221117020158.519218.74675_NLL.20221117170218.53264.116304">
          <pickID>smi:org.gfz-potsdam.de/geofon/Pick/20221117020158.519218.74675</pickID>
          <phase>S</phase>
          <timeCorrection>0.0</timeCorrection>
          <azimuth>121.3101983</azimuth>
          <distance>0.4548760947</distance>
          <takeoffAngle>
            <value>91.9</value>
          </takeoffAngle>
          <timeResidual>-0.5145353243</timeResidual>
          <timeWeight>0.0</timeWeight>
        </arrival>
        <arrival publicID="smi:org.gfz-potsdam.de/geofon/Pick/20221117020158.519584.74677_NLL.20221117170218.53264.116304">
          <pickID>smi:org.gfz-potsdam.de/geofon/Pick/20221117020158.519584.74677</pickID>
          <phase>P</phase>
          <timeCorrection>0.0</timeCorrection>
          <azimuth>11.87053588</azimuth>
          <distance>0.5810370843</distance>
          <takeoffAngle>
            <value>91.4</value>
          </takeoffAngle>
          <timeResidual>-0.2886218026</timeResidual>
          <timeWeight>0.0</timeWeight>
        </arrival>
        <arrival publicID="smi:org.gfz-potsdam.de/geofon/Pick/20221117020158.519658.74679_NLL.20221117170218.53264.116304">
          <pickID>smi:org.gfz-potsdam.de/geofon/Pick/20221117020158.519658.74679</pickID>
          <phase>P</phase>
          <timeCorrection>0.0</timeCorrection>
          <azimuth>53.21059402</azimuth>
          <distance>0.613983072</distance>
          <takeoffAngle>
            <value>91.3</value>
          </takeoffAngle>
          <timeResidual>-0.2930270651</timeResidual>
          <timeWeight>0.0</timeWeight>
        </arrival>
        <arrival publicID="smi:org.gfz-potsdam.de/geofon/Pick/20221117020158.519696.74680_NLL.20221117170218.53264.116304">
          <pickID>smi:org.gfz-potsdam.de/geofon/Pick/20221117020158.519696.74680</pickID>
          <phase>P</phase>
          <timeCorrection>0.0</timeCorrection>
          <azimuth>9.430715816</azimuth>
          <distance>0.63676089</distance>
          <takeoffAngle>
            <value>91.3</value>
          </takeoffAngle>
          <timeResidual>-0.5209884481</timeResidual>
          <timeWeight>0.0</timeWeight>
        </arrival>
        <arrival publicID="smi:org.gfz-potsdam.de/geofon/Pick/20221117020158.519734.74681_NLL.20221117170218.53264.116304">
          <pickID>smi:org.gfz-potsdam.de/geofon/Pick/20221117020158.519734.74681</pickID>
          <phase>P</phase>
          <timeCorrection>0.0</timeCorrection>
          <azimuth>331.5539008</azimuth>
          <distance>0.6445821914</distance>
          <takeoffAngle>
            <value>91.2</value>
          </takeoffAngle>
          <timeResidual>-0.7528582727</timeResidual>
          <timeWeight>0.0</timeWeight>
        </arrival>
        <arrival publicID="smi:org.gfz-potsdam.de/geofon/Pick/20221117020158.51977.74682_NLL.20221117170218.53264.116304">
          <pickID>smi:org.gfz-potsdam.de/geofon/Pick/20221117020158.51977.74682</pickID>
          <phase>P</phase>
          <timeCorrection>0.0</timeCorrection>
          <azimuth>56.66758051</azimuth>
          <distance>0.6516241602</distance>
          <takeoffAngle>
            <value>90.9</value>
          </takeoffAngle>
          <timeResidual>-0.3208085069</timeResidual>
          <timeWeight>0.0</timeWeight>
        </arrival>
        <arrival publicID="smi:org.gfz-potsdam.de/geofon/Pick/20221117020158.519808.74683_NLL.20221117170218.53264.116304">
          <pickID>smi:org.gfz-potsdam.de/geofon/Pick/20221117020158.519808.74683</pickID>
          <phase>P</phase>
          <timeCorrection>0.0</timeCorrection>
          <azimuth>29.719911</azimuth>
          <distance>0.6542330558</distance>
          <takeoffAngle>
            <value>90.9</value>
          </takeoffAngle>
          <timeResidual>-0.3687936719</timeResidual>
          <timeWeight>0.0</timeWeight>
        </arrival>
        <arrival publicID="smi:org.gfz-potsdam.de/geofon/Pick/20221117164826.532236.115001_NLL.20221117170218.53264.116304">
          <pickID>smi:org.gfz-potsdam.de/geofon/Pick/20221117164826.532236.115001</pickID>
          <phase>P</phase>
          <timeCorrection>0.0</timeCorrection>
          <azimuth>84.77310787</azimuth>
          <distance>0.05553466475</distance>
          <takeoffAngle>
            <value>135.9</value>
          </takeoffAngle>
          <timeResidual>0.02816460612</timeResidual>
          <timeWeight>1.262349847</timeWeight>
        </arrival>
        <arrival publicID="smi:org.gfz-potsdam.de/geofon/Pick/20221117164826.5324.115003_NLL.20221117170218.53264.116304">
          <pickID>smi:org.gfz-potsdam.de/geofon/Pick/20221117164826.5324.115003</pickID>
          <phase>P</phase>
          <timeCorrection>0.0</timeCorrection>
          <azimuth>293.6520466</azimuth>
          <distance>0.06873102299</distance>
          <takeoffAngle>
            <value>126.7</value>
          </takeoffAngle>
          <timeResidual>0.09318948877</timeResidual>
          <timeWeight>1.166026176</timeWeight>
        </arrival>
        <arrival publicID="smi:org.gfz-potsdam.de/geofon/Pick/20221117164826.53251.115004_NLL.20221117170218.53264.116304">
          <pickID>smi:org.gfz-potsdam.de/geofon/Pick/20221117164826.53251.115004</pickID>
          <phase>P</phase>
          <timeCorrection>0.0</timeCorrection>
          <azimuth>205.0513223</azimuth>
          <distance>0.1123886009</distance>
          <takeoffAngle>
            <value>108.9</value>
          </takeoffAngle>
          <timeResidual>-0.04855038058</timeResidual>
          <timeWeight>1.230065609</timeWeight>
        </arrival>
        <arrival publicID="smi:org.gfz-potsdam.de/geofon/Pick/20221117164826.532567.115005_NLL.20221117170218.53264.116304">
          <pickID>smi:org.gfz-potsdam.de/geofon/Pick/20221117164826.532567.115005</pickID>
          <phase>S</phase>
          <timeCorrection>0.0</timeCorrection>
          <azimuth>205.0513223</azimuth>
          <distance>0.1123886009</distance>
          <takeoffAngle>
            <value>108.9</value>
          </takeoffAngle>
          <timeResidual>0.1122460104</timeResidual>
          <timeWeight>1.118422886</timeWeight>
        </arrival>
        <arrival publicID="smi:org.gfz-potsdam.de/geofon/Pick/20221117164826.532752.115006_NLL.20221117170218.53264.116304">
          <pickID>smi:org.gfz-potsdam.de/geofon/Pick/20221117164826.532752.115006</pickID>
          <phase>P</phase>
          <timeCorrection>0.0</timeCorrection>
          <azimuth>69.4140313</azimuth>
          <distance>0.1221260299</distance>
          <takeoffAngle>
            <value>106.0</value>
          </takeoffAngle>
          <timeResidual>0.09111202861</timeResidual>
          <timeWeight>1.170635215</timeWeight>
        </arrival>
        <arrival publicID="smi:org.gfz-potsdam.de/geofon/Pick/20221117164826.532803.115007_NLL.20221117170218.53264.116304">
          <pickID>smi:org.gfz-potsdam.de/geofon/Pick/20221117164826.532803.115007</pickID>
          <phase>S</phase>
          <timeCorrection>0.0</timeCorrection>
          <azimuth>69.4140313</azimuth>
          <distance>0.1221260299</distance>
          <takeoffAngle>
            <value>106.0</value>
          </takeoffAngle>
          <timeResidual>0.6583655058</timeResidual>
          <timeWeight>0.0</timeWeight>
        </arrival>
        <arrival publicID="smi:org.gfz-potsdam.de/geofon/Pick/20221117164826.532898.115008_NLL.20221117170218.53264.116304">
          <pickID>smi:org.gfz-potsdam.de/geofon/Pick/20221117164826.532898.115008</pickID>
          <phase>S</phase>
          <timeCorrection>0.0</timeCorrection>
          <azimuth>122.5413655</azimuth>
          <distance>0.1545129781</distance>
          <takeoffAngle>
            <value>99.8</value>
          </takeoffAngle>
          <timeResidual>0.06302936975</timeResidual>
          <timeWeight>0.9752103084</timeWeight>
        </arrival>
        <arrival publicID="smi:org.gfz-potsdam.de/geofon/Pick/20221117164826.533235.115009_NLL.20221117170218.53264.116304">
          <pickID>smi:org.gfz-potsdam.de/geofon/Pick/20221117164826.533235.115009</pickID>
          <phase>P</phase>
          <azimuth>157.0836592</azimuth>
          <distance>0.2890401341</distance>
          <timeWeight>0.0</timeWeight>
        </arrival>
        <arrival publicID="smi:org.gfz-potsdam.de/geofon/Pick/20221117164826.533295.115010_NLL.20221117170218.53264.116304">
          <pickID>smi:org.gfz-potsdam.de/geofon/Pick/20221117164826.533295.115010</pickID>
          <phase>S</phase>
          <azimuth>157.0836592</azimuth>
          <distance>0.2890401341</distance>
          <timeWeight>0.0</timeWeight>
        </arrival>
        <arrival publicID="smi:org.gfz-potsdam.de/geofon/Pick/20221117164826.533321.115011_NLL.20221117170218.53264.116304">
          <pickID>smi:org.gfz-potsdam.de/geofon/Pick/20221117164826.533321.115011</pickID>
          <phase>S</phase>
          <timeCorrection>0.0</timeCorrection>
          <azimuth>101.6458459</azimuth>
          <distance>0.2880350502</distance>
          <takeoffAngle>
            <value>93.4</value>
          </takeoffAngle>
          <timeResidual>0.2175206356</timeResidual>
          <timeWeight>0.7438256294</timeWeight>
        </arrival>
        <arrival publicID="smi:org.gfz-potsdam.de/geofon/Pick/20221117164826.533355.115012_NLL.20221117170218.53264.116304">
          <pickID>smi:org.gfz-potsdam.de/geofon/Pick/20221117164826.533355.115012</pickID>
          <phase>P</phase>
          <timeCorrection>0.0</timeCorrection>
          <azimuth>101.6458459</azimuth>
          <distance>0.2880350502</distance>
          <takeoffAngle>
            <value>93.4</value>
          </takeoffAngle>
          <timeResidual>0.0449205902</timeResidual>
          <timeWeight>0.98787902</timeWeight>
        </arrival>
        <arrival publicID="smi:org.gfz-potsdam.de/geofon/Pick/20221117165346.006932.115338_NLL.20221117170218.53264.116304">
          <pickID>smi:org.gfz-potsdam.de/geofon/Pick/20221117165346.006932.115338</pickID>
          <phase>S</phase>
          <timeCorrection>0.0</timeCorrection>
          <azimuth>84.77310787</azimuth>
          <distance>0.05553466475</distance>
          <takeoffAngle>
            <value>135.9</value>
          </takeoffAngle>
          <timeResidual>-0.2025323301</timeResidual>
          <timeWeight>0.7898652033</timeWeight>
        </arrival>
        <arrival publicID="smi:org.gfz-potsdam.de/geofon/Pick/20221117165346.007624.115339_NLL.20221117170218.53264.116304">
          <pickID>smi:org.gfz-potsdam.de/geofon/Pick/20221117165346.007624.115339</pickID>
          <phase>P</phase>
          <azimuth>354.2692406</azimuth>
          <distance>0.3002904793</distance>
          <timeWeight>0.0</timeWeight>
        </arrival>
        <arrival publicID="smi:org.gfz-potsdam.de/geofon/Pick/20221117165346.00767.115340_NLL.20221117170218.53264.116304">
          <pickID>smi:org.gfz-potsdam.de/geofon/Pick/20221117165346.00767.115340</pickID>
          <phase>S</phase>
          <azimuth>354.2692406</azimuth>
          <distance>0.3002904793</distance>
          <timeWeight>0.0</timeWeight>
        </arrival>
        <arrival publicID="smi:org.gfz-potsdam.de/geofon/Pick/20221117165346.007912.115341_NLL.20221117170218.53264.116304">
          <pickID>smi:org.gfz-potsdam.de/geofon/Pick/20221117165346.007912.115341</pickID>
          <phase>P</phase>
          <timeCorrection>0.0</timeCorrection>
          <azimuth>277.171844</azimuth>
          <distance>0.3667703823</distance>
          <takeoffAngle>
            <value>92.5</value>
          </takeoffAngle>
          <timeResidual>-0.2321942477</timeResidual>
          <timeWeight>0.0</timeWeight>
        </arrival>
        <arrival publicID="smi:org.gfz-potsdam.de/geofon/Pick/20221117165346.00795.115342_NLL.20221117170218.53264.116304">
          <pickID>smi:org.gfz-potsdam.de/geofon/Pick/20221117165346.00795.115342</pickID>
          <phase>S</phase>
          <timeCorrection>0.0</timeCorrection>
          <azimuth>277.171844</azimuth>
          <distance>0.3667703823</distance>
          <takeoffAngle>
            <value>92.5</value>
          </takeoffAngle>
          <timeResidual>-1.076563304</timeResidual>
          <timeWeight>0.0</timeWeight>
        </arrival>
        <arrival publicID="smi:org.gfz-potsdam.de/geofon/Pick/20221117165747.528372.115574_NLL.20221117170218.53264.116304">
          <pickID>smi:org.gfz-potsdam.de/geofon/Pick/20221117165747.528372.115574</pickID>
          <phase>S</phase>
          <timeCorrection>0.0</timeCorrection>
          <azimuth>117.1958352</azimuth>
          <distance>0.4932828739</distance>
          <takeoffAngle>
            <value>91.7</value>
          </takeoffAngle>
          <timeResidual>-0.2476545206</timeResidual>
          <timeWeight>0.5888862895</timeWeight>
        </arrival>
        <arrival publicID="smi:org.gfz-potsdam.de/geofon/Pick/20221117165747.528533.115575_NLL.20221117170218.53264.116304">
          <pickID>smi:org.gfz-potsdam.de/geofon/Pick/20221117165747.528533.115575</pickID>
          <phase>S</phase>
          <timeCorrection>0.0</timeCorrection>
          <azimuth>152.4515032</azimuth>
          <distance>0.5048801691</distance>
          <takeoffAngle>
            <value>91.7</value>
          </takeoffAngle>
          <timeResidual>-1.133568566</timeResidual>
          <timeWeight>0.0</timeWeight>
        </arrival>
        <arrival publicID="smi:org.gfz-potsdam.de/geofon/Pick/20221117165747.52863.115576_NLL.20221117170218.53264.116304">
          <pickID>smi:org.gfz-potsdam.de/geofon/Pick/20221117165747.52863.115576</pickID>
          <phase>S</phase>
          <timeCorrection>0.0</timeCorrection>
          <azimuth>119.5868922</azimuth>
          <distance>0.5048911493</distance>
          <takeoffAngle>
            <value>91.7</value>
          </takeoffAngle>
          <timeResidual>-0.4175266671</timeResidual>
          <timeWeight>0.0</timeWeight>
        </arrival>
        <arrival publicID="smi:org.gfz-potsdam.de/geofon/Pick/20221117165747.528694.115577_NLL.20221117170218.53264.116304">
          <pickID>smi:org.gfz-potsdam.de/geofon/Pick/20221117165747.528694.115577</pickID>
          <phase>S</phase>
          <timeCorrection>0.0</timeCorrection>
          <azimuth>42.98725227</azimuth>
          <distance>0.5033884852</distance>
          <takeoffAngle>
            <value>91.7</value>
          </takeoffAngle>
          <timeResidual>-0.6388785779</timeResidual>
          <timeWeight>0.0</timeWeight>
        </arrival>
        <arrival publicID="smi:org.gfz-potsdam.de/geofon/Pick/20221117165747.528778.115578_NLL.20221117170218.53264.116304">
          <pickID>smi:org.gfz-potsdam.de/geofon/Pick/20221117165747.528778.115578</pickID>
          <phase>S</phase>
          <timeCorrection>0.0</timeCorrection>
          <azimuth>136.2029496</azimuth>
          <distance>0.5101243886</distance>
          <takeoffAngle>
            <value>91.7</value>
          </takeoffAngle>
          <timeResidual>-0.5325629428</timeResidual>
          <timeWeight>0.0</timeWeight>
        </arrival>
        <arrival publicID="smi:org.gfz-potsdam.de/geofon/Pick/20221117165747.528842.115579_NLL.20221117170218.53264.116304">
          <pickID>smi:org.gfz-potsdam.de/geofon/Pick/20221117165747.528842.115579</pickID>
          <phase>S</phase>
          <timeCorrection>0.0</timeCorrection>
          <azimuth>142.324561</azimuth>
          <distance>0.5141647447</distance>
          <takeoffAngle>
            <value>91.7</value>
          </takeoffAngle>
          <timeResidual>-0.5449458613</timeResidual>
          <timeWeight>0.0</timeWeight>
        </arrival>
        <arrival publicID="smi:org.gfz-potsdam.de/geofon/Pick/20221117165747.528881.115580_NLL.20221117170218.53264.116304">
          <pickID>smi:org.gfz-potsdam.de/geofon/Pick/20221117165747.528881.115580</pickID>
          <phase>S</phase>
          <timeCorrection>0.0</timeCorrection>
          <azimuth>38.23757938</azimuth>
          <distance>0.5543872717</distance>
          <takeoffAngle>
            <value>91.5</value>
          </takeoffAngle>
          <timeResidual>-0.3728774883</timeResidual>
          <timeWeight>0.0</timeWeight>
        </arrival>
        <arrival publicID="smi:org.gfz-potsdam.de/geofon/Pick/20221117165747.52892.115581_NLL.20221117170218.53264.116304">
          <pickID>smi:org.gfz-potsdam.de/geofon/Pick/20221117165747.52892.115581</pickID>
          <phase>P</phase>
          <timeCorrection>0.0</timeCorrection>
          <azimuth>38.23757938</azimuth>
          <distance>0.5543872717</distance>
          <takeoffAngle>
            <value>91.5</value>
          </takeoffAngle>
          <timeResidual>-0.2562345191</timeResidual>
          <timeWeight>0.0</timeWeight>
        </arrival>
        <arrival publicID="smi:org.gfz-potsdam.de/geofon/Pick/20221117165747.528958.115582_NLL.20221117170218.53264.116304">
          <pickID>smi:org.gfz-potsdam.de/geofon/Pick/20221117165747.528958.115582</pickID>
          <phase>S</phase>
          <timeCorrection>0.0</timeCorrection>
          <azimuth>17.58941913</azimuth>
          <distance>0.5567573036</distance>
          <takeoffAngle>
            <value>91.5</value>
          </takeoffAngle>
          <timeResidual>-0.03960146361</timeResidual>
          <timeWeight>0.6997124295</timeWeight>
        </arrival>
        <arrival publicID="smi:org.gfz-potsdam.de/geofon/Pick/20221117165747.528997.115583_NLL.20221117170218.53264.116304">
          <pickID>smi:org.gfz-potsdam.de/geofon/Pick/20221117165747.528997.115583</pickID>
          <phase>P</phase>
          <timeCorrection>0.0</timeCorrection>
          <azimuth>17.58941913</azimuth>
          <distance>0.5567573036</distance>
          <takeoffAngle>
            <value>91.5</value>
          </takeoffAngle>
          <timeResidual>-0.2370754947</timeResidual>
          <timeWeight>0.0</timeWeight>
        </arrival>
        <arrival publicID="smi:org.gfz-potsdam.de/geofon/Pick/20221117165747.529072.115584_NLL.20221117170218.53264.116304">
          <pickID>smi:org.gfz-potsdam.de/geofon/Pick/20221117165747.529072.115584</pickID>
          <phase>S</phase>
          <timeCorrection>0.0</timeCorrection>
          <azimuth>125.025231</azimuth>
          <distance>0.6110330187</distance>
          <takeoffAngle>
            <value>91.3</value>
          </takeoffAngle>
          <timeResidual>-1.075925311</timeResidual>
          <timeWeight>0.0</timeWeight>
        </arrival>
        <arrival publicID="smi:org.gfz-potsdam.de/geofon/Pick/20221117165747.529145.115585_NLL.20221117170218.53264.116304">
          <pickID>smi:org.gfz-potsdam.de/geofon/Pick/20221117165747.529145.115585</pickID>
          <phase>S</phase>
          <timeCorrection>0.0</timeCorrection>
          <azimuth>53.21059402</azimuth>
          <distance>0.613983072</distance>
          <takeoffAngle>
            <value>91.3</value>
          </takeoffAngle>
          <timeResidual>-0.9012478082</timeResidual>
          <timeWeight>0.0</timeWeight>
        </arrival>
        <arrival publicID="smi:org.gfz-potsdam.de/geofon/Pick/20221117165747.529448.115586_NLL.20221117170218.53264.116304">
          <pickID>smi:org.gfz-potsdam.de/geofon/Pick/20221117165747.529448.115586</pickID>
          <phase>P</phase>
          <timeCorrection>0.0</timeCorrection>
          <azimuth>141.2690942</azimuth>
          <distance>0.9212758196</distance>
          <takeoffAngle>
            <value>77.5</value>
          </takeoffAngle>
          <timeResidual>-0.7619020893</timeResidual>
          <timeWeight>0.0</timeWeight>
        </arrival>
        <arrival publicID="smi:org.gfz-potsdam.de/geofon/Pick/20221117165747.529487.115587_NLL.20221117170218.53264.116304">
          <pickID>smi:org.gfz-potsdam.de/geofon/Pick/20221117165747.529487.115587</pickID>
          <phase>S</phase>
          <timeCorrection>0.0</timeCorrection>
          <azimuth>141.2690942</azimuth>
          <distance>0.9212758196</distance>
          <takeoffAngle>
            <value>77.5</value>
          </takeoffAngle>
          <timeResidual>-1.673355182</timeResidual>
          <timeWeight>0.0</timeWeight>
        </arrival>
        <arrival publicID="smi:org.gfz-potsdam.de/geofon/Pick/20221117165747.529527.115588_NLL.20221117170218.53264.116304">
          <pickID>smi:org.gfz-potsdam.de/geofon/Pick/20221117165747.529527.115588</pickID>
          <phase>P</phase>
          <timeCorrection>0.0</timeCorrection>
          <azimuth>123.4998097</azimuth>
          <distance>0.9444318461</distance>
          <takeoffAngle>
            <value>77.5</value>
          </takeoffAngle>
          <timeResidual>-0.6061937544</timeResidual>
          <timeWeight>0.0</timeWeight>
        </arrival>
        <arrival publicID="smi:org.gfz-potsdam.de/geofon/Pick/20221117165747.52956.115589_NLL.20221117170218.53264.116304">
          <pickID>smi:org.gfz-potsdam.de/geofon/Pick/20221117165747.52956.115589</pickID>
          <phase>S</phase>
          <timeCorrection>0.0</timeCorrection>
          <azimuth>123.4998097</azimuth>
          <distance>0.9444318461</distance>
          <takeoffAngle>
            <value>77.5</value>
          </takeoffAngle>
          <timeResidual>-1.237554009</timeResidual>
          <timeWeight>0.0</timeWeight>
        </arrival>
        <arrival publicID="smi:org.gfz-potsdam.de/geofon/Pick/20221117165747.529606.115590_NLL.20221117170218.53264.116304">
          <pickID>smi:org.gfz-potsdam.de/geofon/Pick/20221117165747.529606.115590</pickID>
          <phase>P</phase>
          <timeCorrection>0.0</timeCorrection>
          <azimuth>116.6374362</azimuth>
          <distance>0.9486455017</distance>
          <takeoffAngle>
            <value>77.5</value>
          </takeoffAngle>
          <timeResidual>-1.5850591</timeResidual>
          <timeWeight>0.0</timeWeight>
        </arrival>
        <arrival publicID="smi:org.gfz-potsdam.de/geofon/Pick/20221117165747.529639.115591_NLL.20221117170218.53264.116304">
          <pickID>smi:org.gfz-potsdam.de/geofon/Pick/20221117165747.529639.115591</pickID>
          <phase>S</phase>
          <timeCorrection>0.0</timeCorrection>
          <azimuth>116.6374362</azimuth>
          <distance>0.9486455017</distance>
          <takeoffAngle>
            <value>77.5</value>
          </takeoffAngle>
          <timeResidual>-1.522783098</timeResidual>
          <timeWeight>0.0</timeWeight>
        </arrival>
        <arrival publicID="smi:org.gfz-potsdam.de/geofon/Pick/20221117165747.52968.115592_NLL.20221117170218.53264.116304">
          <pickID>smi:org.gfz-potsdam.de/geofon/Pick/20221117165747.52968.115592</pickID>
          <phase>P</phase>
          <timeCorrection>0.0</timeCorrection>
          <azimuth>126.7764126</azimuth>
          <distance>0.9541560896</distance>
          <takeoffAngle>
            <value>77.5</value>
          </takeoffAngle>
          <timeResidual>-0.6302403846</timeResidual>
          <timeWeight>0.0</timeWeight>
        </arrival>
      </origin>
      <magnitude publicID="smi:org.gfz-potsdam.de/geofon/Magnitude/20221117170222.739536.116338">
        <mag>
          <value>5.432978086</value>
          <uncertainty>0.155090643</uncertainty>
        </mag>
        <type>ML(TexNet)</type>
        <originID>smi:org.gfz-potsdam.de/geofon/NLL.20221117170218.53264.116304</originID>
        <methodID>smi:org.gfz-potsdam.de/geofon/trimmed_mean</methodID>
        <stationCount>21</stationCount>
        <evaluationStatus>confirmed</evaluationStatus>
        <creationInfo>
          <agencyID>TXNet</agencyID>
          <author>daniel</author>
          <creationTime>2022-11-17T17:09:30.664136Z</creationTime>
        </creationInfo>
      </magnitude>
      <pick publicID="smi:org.gfz-potsdam.de/geofon/Pick/20221116213904.882278.81130">
        <time>
          <value>2022-11-16T21:33:00.410982Z</value>
          <lowerUncertainty>0.3000000119</lowerUncertainty>
          <upperUncertainty>0.3000000119</upperUncertainty>
        </time>
        <waveformID networkCode="TX" stationCode="DG07" locationCode="00" channelCode="CH1"></waveformID>
        <phaseHint>S</phaseHint>
        <evaluationMode>manual</evaluationMode>
        <creationInfo>
          <agencyID>TXNet</agencyID>
          <author>cooper</author>
          <creationTime>2022-11-16T21:39:04.882318Z</creationTime>
        </creationInfo>
      </pick>
      <pick publicID="smi:org.gfz-potsdam.de/geofon/Pick/20221116213904.88235.81131">
        <time>
          <value>2022-11-16T21:33:01.245419Z</value>
          <lowerUncertainty>0.3000000119</lowerUncertainty>
          <upperUncertainty>0.3000000119</upperUncertainty>
        </time>
        <waveformID networkCode="TX" stationCode="PCOS" locationCode="00" channelCode="HHE"></waveformID>
        <phaseHint>S</phaseHint>
        <evaluationMode>manual</evaluationMode>
        <creationInfo>
          <agencyID>TXNet</agencyID>
          <author>cooper</author>
          <creationTime>2022-11-16T21:39:04.882386Z</creationTime>
        </creationInfo>
      </pick>
      <pick publicID="smi:org.gfz-potsdam.de/geofon/Pick/20221116214756.54407.84373">
        <time>
          <value>2022-11-16T21:32:46.649570Z</value>
          <lowerUncertainty>0.1000000015</lowerUncertainty>
          <upperUncertainty>0.1000000015</upperUncertainty>
        </time>
        <waveformID networkCode="TX" stationCode="PB34" locationCode="00" channelCode="HHZ"></waveformID>
        <phaseHint>P</phaseHint>
        <evaluationMode>manual</evaluationMode>
        <creationInfo>
          <agencyID>TXNet</agencyID>
          <author>cooper</author>
          <creationTime>2022-11-16T21:47:56.544213Z</creationTime>
        </creationInfo>
      </pick>
      <pick publicID="smi:org.gfz-potsdam.de/geofon/Pick/20221116214756.544252.84374">
        <time>
          <value>2022-11-16T21:32:47.948153Z</value>
          <lowerUncertainty>0.200000003</lowerUncertainty>
          <upperUncertainty>0.200000003</upperUncertainty>
        </time>
        <waveformID networkCode="TX" stationCode="PB34" locationCode="00" channelCode="HH1"></waveformID>
        <phaseHint>S</phaseHint>
        <evaluationMode>manual</evaluationMode>
        <creationInfo>
          <agencyID>TXNet</agencyID>
          <author>cooper</author>
          <creationTime>2022-11-16T21:47:56.544283Z</creationTime>
        </creationInfo>
      </pick>
      <pick publicID="smi:org.gfz-potsdam.de/geofon/Pick/20221116214756.544409.84377">
        <time>
          <value>2022-11-16T21:32:47.360699Z</value>
          <lowerUncertainty>0.200000003</lowerUncertainty>
          <upperUncertainty>0.200000003</upperUncertainty>
        </time>
        <waveformID networkCode="4O" stationCode="WB01" locationCode="00" channelCode="HHZ"></waveformID>
        <phaseHint>P</phaseHint>
        <evaluationMode>manual</evaluationMode>
        <creationInfo>
          <agencyID>TXNet</agencyID>
          <author>cooper</author>
          <creationTime>2022-11-16T21:47:56.544435Z</creationTime>
        </creationInfo>
      </pick>
      <pick publicID="smi:org.gfz-potsdam.de/geofon/Pick/20221116214756.544453.84378">
        <time>
          <value>2022-11-16T21:32:49.277653Z</value>
          <lowerUncertainty>0.200000003</lowerUncertainty>
          <upperUncertainty>0.200000003</upperUncertainty>
        </time>
        <waveformID networkCode="4O" stationCode="WB01" locationCode="00" channelCode="HHN"></waveformID>
        <phaseHint>S</phaseHint>
        <evaluationMode>manual</evaluationMode>
        <creationInfo>
          <agencyID>TXNet</agencyID>
          <author>cooper</author>
          <creationTime>2022-11-16T21:47:56.544478Z</creationTime>
        </creationInfo>
      </pick>
      <pick publicID="smi:org.gfz-potsdam.de/geofon/Pick/20221116214756.544502.84379">
        <time>
          <value>2022-11-16T21:32:51.101852Z</value>
          <lowerUncertainty>0.3000000119</lowerUncertainty>
          <upperUncertainty>0.3000000119</upperUncertainty>
        </time>
        <waveformID networkCode="4O" stationCode="SA06" locationCode="00" channelCode="HHE"></waveformID>
        <filterID>smi:org.gfz-potsdam.de/geofon/BW(2,2,8)</filterID>
        <phaseHint>S</phaseHint>
        <evaluationMode>manual</evaluationMode>
        <creationInfo>
          <agencyID>TXNet</agencyID>
          <author>cooper</author>
          <creationTime>2022-11-16T21:47:56.544527Z</creationTime>
        </creationInfo>
      </pick>
      <pick publicID="smi:org.gfz-potsdam.de/geofon/Pick/20221116214756.544546.84380">
        <time>
          <value>2022-11-16T21:32:48.697929Z</value>
          <lowerUncertainty>0.200000003</lowerUncertainty>
          <upperUncertainty>0.200000003</upperUncertainty>
        </time>
        <waveformID networkCode="4O" stationCode="SA06" locationCode="00" channelCode="HHZ"></waveformID>
        <filterID>smi:org.gfz-potsdam.de/geofon/BW(2,2,8)</filterID>
        <phaseHint>P</phaseHint>
        <evaluationMode>manual</evaluationMode>
        <creationInfo>
          <agencyID>TXNet</agencyID>
          <author>cooper</author>
          <creationTime>2022-11-16T21:47:56.544572Z</creationTime>
        </creationInfo>
      </pick>
      <pick publicID="smi:org.gfz-potsdam.de/geofon/Pick/20221116214756.544593.84381">
        <time>
          <value>2022-11-16T21:32:48.891171Z</value>
          <lowerUncertainty>0.200000003</lowerUncertainty>
          <upperUncertainty>0.200000003</upperUncertainty>
        </time>
        <waveformID networkCode="4O" stationCode="WB05" locationCode="00" channelCode="HHZ"></waveformID>
        <filterID>smi:org.gfz-potsdam.de/geofon/BW(2,2,8)</filterID>
        <phaseHint>P</phaseHint>
        <evaluationMode>manual</evaluationMode>
        <creationInfo>
          <agencyID>TXNet</agencyID>
          <author>cooper</author>
          <creationTime>2022-11-16T21:47:56.544619Z</creationTime>
        </creationInfo>
      </pick>
      <pick publicID="smi:org.gfz-potsdam.de/geofon/Pick/20221116214756.544638.84382">
        <time>
          <value>2022-11-16T21:32:52.075791Z</value>
          <lowerUncertainty>0.3000000119</lowerUncertainty>
          <upperUncertainty>0.3000000119</upperUncertainty>
        </time>
        <waveformID networkCode="4O" stationCode="WB05" locationCode="00" channelCode="HHN"></waveformID>
        <phaseHint>S</phaseHint>
        <evaluationMode>manual</evaluationMode>
        <creationInfo>
          <agencyID>TXNet</agencyID>
          <author>cooper</author>
          <creationTime>2022-11-16T21:47:56.544662Z</creationTime>
        </creationInfo>
      </pick>
      <pick publicID="smi:org.gfz-potsdam.de/geofon/Pick/20221116214756.544684.84383">
        <time>
          <value>2022-11-16T21:32:49.725976Z</value>
          <lowerUncertainty>0.3000000119</lowerUncertainty>
          <upperUncertainty>0.3000000119</upperUncertainty>
        </time>
        <waveformID networkCode="4O" stationCode="SA02" locationCode="00" channelCode="HHZ"></waveformID>
        <phaseHint>P</phaseHint>
        <evaluationMode>manual</evaluationMode>
        <creationInfo>
          <agencyID>TXNet</agencyID>
          <author>cooper</author>
          <creationTime>2022-11-16T21:47:56.544710Z</creationTime>
        </creationInfo>
      </pick>
      <pick publicID="smi:org.gfz-potsdam.de/geofon/Pick/20221116215249.515307.86784">
        <time>
          <value>2022-11-16T21:32:52.774102Z</value>
        </time>
        <waveformID networkCode="4O" stationCode="SA02" locationCode="00" channelCode="HHE"></waveformID>
        <phaseHint>S</phaseHint>
        <evaluationMode>manual</evaluationMode>
        <creationInfo>
          <agencyID>TXNet</agencyID>
          <author>cooper</author>
          <creationTime>2022-11-16T21:52:49.515437Z</creationTime>
        </creationInfo>
      </pick>
      <pick publicID="smi:org.gfz-potsdam.de/geofon/Pick/20221116215249.51548.86785">
        <time>
          <value>2022-11-16T21:32:53.504887Z</value>
          <lowerUncertainty>0.1000000015</lowerUncertainty>
          <upperUncertainty>0.1000000015</upperUncertainty>
        </time>
        <waveformID networkCode="4O" stationCode="SA04" locationCode="00" channelCode="HHN"></waveformID>
        <phaseHint>S</phaseHint>
        <evaluationMode>manual</evaluationMode>
        <creationInfo>
          <agencyID>TXNet</agencyID>
          <author>cooper</author>
          <creationTime>2022-11-16T21:52:49.515518Z</creationTime>
        </creationInfo>
      </pick>
      <pick publicID="smi:org.gfz-potsdam.de/geofon/Pick/20221116215249.515542.86786">
        <time>
          <value>2022-11-16T21:32:49.922607Z</value>
          <lowerUncertainty>0.1000000015</lowerUncertainty>
          <upperUncertainty>0.1000000015</upperUncertainty>
        </time>
        <waveformID networkCode="4O" stationCode="SA04" locationCode="00" channelCode="HHN"></waveformID>
        <phaseHint>P</phaseHint>
        <evaluationMode>manual</evaluationMode>
        <creationInfo>
          <agencyID>TXNet</agencyID>
          <author>cooper</author>
          <creationTime>2022-11-16T21:52:49.515575Z</creationTime>
        </creationInfo>
      </pick>
      <pick publicID="smi:org.gfz-potsdam.de/geofon/Pick/20221116215249.515612.86787">
        <time>
          <value>2022-11-16T21:32:54.393292Z</value>
          <lowerUncertainty>0.1000000015</lowerUncertainty>
          <upperUncertainty>0.1000000015</upperUncertainty>
        </time>
        <waveformID networkCode="TX" stationCode="PECS" locationCode="00" channelCode="HH1"></waveformID>
        <phaseHint>S</phaseHint>
        <evaluationMode>manual</evaluationMode>
        <creationInfo>
          <agencyID>TXNet</agencyID>
          <author>cooper</author>
          <creationTime>2022-11-16T21:52:49.515646Z</creationTime>
        </creationInfo>
      </pick>
      <pick publicID="smi:org.gfz-potsdam.de/geofon/Pick/20221116215249.515669.86788">
        <time>
          <value>2022-11-16T21:32:50.323823Z</value>
          <lowerUncertainty>0.1000000015</lowerUncertainty>
          <upperUncertainty>0.1000000015</upperUncertainty>
        </time>
        <waveformID networkCode="TX" stationCode="PECS" locationCode="00" channelCode="HH1"></waveformID>
        <phaseHint>P</phaseHint>
        <evaluationMode>manual</evaluationMode>
        <creationInfo>
          <agencyID>TXNet</agencyID>
          <author>cooper</author>
          <creationTime>2022-11-16T21:52:49.515700Z</creationTime>
        </creationInfo>
      </pick>
      <pick publicID="smi:org.gfz-potsdam.de/geofon/Pick/20221116215249.515835.86791">
        <time>
          <value>2022-11-16T21:32:52.673798Z</value>
        </time>
        <waveformID networkCode="TX" stationCode="PB10" locationCode="00" channelCode="HH1"></waveformID>
        <phaseHint>P</phaseHint>
        <evaluationMode>manual</evaluationMode>
        <creationInfo>
          <agencyID>TXNet</agencyID>
          <author>cooper</author>
          <creationTime>2022-11-16T21:52:49.515867Z</creationTime>
        </creationInfo>
      </pick>
      <pick publicID="smi:org.gfz-potsdam.de/geofon/Pick/20221116215249.515889.86792">
        <time>
          <value>2022-11-16T21:32:58.720685Z</value>
          <lowerUncertainty>0.200000003</lowerUncertainty>
          <upperUncertainty>0.200000003</upperUncertainty>
        </time>
        <waveformID networkCode="TX" stationCode="PB10" locationCode="00" channelCode="HH1"></waveformID>
        <phaseHint>S</phaseHint>
        <evaluationMode>manual</evaluationMode>
        <creationInfo>
          <agencyID>TXNet</agencyID>
          <author>cooper</author>
          <creationTime>2022-11-16T21:52:49.515919Z</creationTime>
        </creationInfo>
      </pick>
      <pick publicID="smi:org.gfz-potsdam.de/geofon/Pick/20221116215249.515945.86793">
        <time>
          <value>2022-11-16T21:32:52.888735Z</value>
          <lowerUncertainty>0.1000000015</lowerUncertainty>
          <upperUncertainty>0.1000000015</upperUncertainty>
        </time>
        <waveformID networkCode="TX" stationCode="PB12" locationCode="00" channelCode="HHN"></waveformID>
        <phaseHint>P</phaseHint>
        <evaluationMode>manual</evaluationMode>
        <creationInfo>
          <agencyID>TXNet</agencyID>
          <author>cooper</author>
          <creationTime>2022-11-16T21:52:49.515975Z</creationTime>
        </creationInfo>
      </pick>
      <pick publicID="smi:org.gfz-potsdam.de/geofon/Pick/20221116215249.516053.86795">
        <time>
          <value>2022-11-16T21:32:53.132330Z</value>
        </time>
        <waveformID networkCode="TX" stationCode="PB28" locationCode="00" channelCode="HHZ"></waveformID>
        <phaseHint>P</phaseHint>
        <evaluationMode>manual</evaluationMode>
        <creationInfo>
          <agencyID>TXNet</agencyID>
          <author>cooper</author>
          <creationTime>2022-11-16T21:52:49.516086Z</creationTime>
        </creationInfo>
      </pick>
      <pick publicID="smi:org.gfz-potsdam.de/geofon/Pick/20221116215249.516168.86797">
        <time>
          <value>2022-11-16T21:32:54.694203Z</value>
          <lowerUncertainty>0.200000003</lowerUncertainty>
          <upperUncertainty>0.200000003</upperUncertainty>
        </time>
        <waveformID networkCode="4O" stationCode="LWM1" locationCode="0K" channelCode="HHZ"></waveformID>
        <phaseHint>P</phaseHint>
        <evaluationMode>manual</evaluationMode>
        <creationInfo>
          <agencyID>TXNet</agencyID>
          <author>cooper</author>
          <creationTime>2022-11-16T21:52:49.516199Z</creationTime>
        </creationInfo>
      </pick>
      <pick publicID="smi:org.gfz-potsdam.de/geofon/Pick/20221116215249.516222.86798">
        <time>
          <value>2022-11-16T21:32:57.574356Z</value>
          <lowerUncertainty>0.1000000015</lowerUncertainty>
          <upperUncertainty>0.1000000015</upperUncertainty>
        </time>
        <waveformID networkCode="TX" stationCode="PB06" locationCode="00" channelCode="HHZ"></waveformID>
        <phaseHint>P</phaseHint>
        <evaluationMode>manual</evaluationMode>
        <creationInfo>
          <agencyID>TXNet</agencyID>
          <author>cooper</author>
          <creationTime>2022-11-16T21:52:49.516252Z</creationTime>
        </creationInfo>
      </pick>
      <pick publicID="smi:org.gfz-potsdam.de/geofon/Pick/20221116215249.516274.86799">
        <time>
          <value>2022-11-16T21:33:06.186155Z</value>
          <lowerUncertainty>0.200000003</lowerUncertainty>
          <upperUncertainty>0.200000003</upperUncertainty>
        </time>
        <waveformID networkCode="TX" stationCode="PB06" locationCode="00" channelCode="HH1"></waveformID>
        <phaseHint>S</phaseHint>
        <evaluationMode>manual</evaluationMode>
        <creationInfo>
          <agencyID>TXNet</agencyID>
          <author>cooper</author>
          <creationTime>2022-11-16T21:52:49.516303Z</creationTime>
        </creationInfo>
      </pick>
      <pick publicID="smi:org.gfz-potsdam.de/geofon/Pick/20221116215249.516327.86800">
        <time>
          <value>2022-11-16T21:32:57.975571Z</value>
          <lowerUncertainty>0.1000000015</lowerUncertainty>
          <upperUncertainty>0.1000000015</upperUncertainty>
        </time>
        <waveformID networkCode="TX" stationCode="PB30" locationCode="00" channelCode="HHZ"></waveformID>
        <phaseHint>P</phaseHint>
        <polarity>positive</polarity>
        <evaluationMode>manual</evaluationMode>
        <creationInfo>
          <agencyID>TXNet</agencyID>
          <author>cooper</author>
          <creationTime>2022-11-16T21:52:49.516357Z</creationTime>
        </creationInfo>
      </pick>
      <pick publicID="smi:org.gfz-potsdam.de/geofon/Pick/20221116215249.516374.86801">
        <time>
          <value>2022-11-16T21:33:07.203522Z</value>
          <lowerUncertainty>0.5</lowerUncertainty>
          <upperUncertainty>0.5</upperUncertainty>
        </time>
        <waveformID networkCode="TX" stationCode="PB30" locationCode="00" channelCode="HHN"></waveformID>
        <phaseHint>S</phaseHint>
        <evaluationMode>manual</evaluationMode>
        <creationInfo>
          <agencyID>TXNet</agencyID>
          <author>cooper</author>
          <creationTime>2022-11-16T21:52:49.516403Z</creationTime>
        </creationInfo>
      </pick>
      <pick publicID="smi:org.gfz-potsdam.de/geofon/Pick/20221116215249.516427.86802">
        <time>
          <value>2022-11-16T21:32:57.889596Z</value>
        </time>
        <waveformID networkCode="TX" stationCode="PB03" locationCode="00" channelCode="HHZ"></waveformID>
        <phaseHint>P</phaseHint>
        <evaluationMode>manual</evaluationMode>
        <creationInfo>
          <agencyID>TXNet</agencyID>
          <author>cooper</author>
          <creationTime>2022-11-16T21:52:49.516458Z</creationTime>
        </creationInfo>
      </pick>
      <pick publicID="smi:org.gfz-potsdam.de/geofon/Pick/20221116215249.516479.86803">
        <time>
          <value>2022-11-16T21:33:07.303826Z</value>
        </time>
        <waveformID networkCode="TX" stationCode="PB03" locationCode="00" channelCode="HH1"></waveformID>
        <phaseHint>S</phaseHint>
        <evaluationMode>manual</evaluationMode>
        <creationInfo>
          <agencyID>TXNet</agencyID>
          <author>cooper</author>
          <creationTime>2022-11-16T21:52:49.516508Z</creationTime>
        </creationInfo>
      </pick>
      <pick publicID="smi:org.gfz-potsdam.de/geofon/Pick/20221116215503.577442.88046">
        <time>
          <value>2022-11-16T21:32:53.756812Z</value>
        </time>
        <waveformID networkCode="TX" stationCode="DG07" locationCode="00" channelCode="CHZ"></waveformID>
        <phaseHint>P</phaseHint>
        <evaluationMode>manual</evaluationMode>
        <creationInfo>
          <agencyID>TXNet</agencyID>
          <author>cooper</author>
          <creationTime>2022-11-16T21:55:03.577578Z</creationTime>
        </creationInfo>
      </pick>
      <pick publicID="smi:org.gfz-potsdam.de/geofon/Pick/20221116215503.577625.88047">
        <time>
          <value>2022-11-16T21:32:54.258331Z</value>
        </time>
        <waveformID networkCode="TX" stationCode="PCOS" locationCode="00" channelCode="HHN"></waveformID>
        <phaseHint>P</phaseHint>
        <evaluationMode>manual</evaluationMode>
        <creationInfo>
          <agencyID>TXNet</agencyID>
          <author>cooper</author>
          <creationTime>2022-11-16T21:55:03.577665Z</creationTime>
        </creationInfo>
      </pick>
      <pick publicID="smi:org.gfz-potsdam.de/geofon/Pick/20221116215503.577692.88048">
        <time>
          <value>2022-11-16T21:32:54.659546Z</value>
        </time>
        <waveformID networkCode="4O" stationCode="LWM2" locationCode="0K" channelCode="HHZ"></waveformID>
        <phaseHint>P</phaseHint>
        <evaluationMode>manual</evaluationMode>
        <creationInfo>
          <agencyID>TXNet</agencyID>
          <author>cooper</author>
          <creationTime>2022-11-16T21:55:03.577728Z</creationTime>
        </creationInfo>
      </pick>
      <pick publicID="smi:org.gfz-potsdam.de/geofon/Pick/20221117015938.385144.74510">
        <time>
          <value>2022-11-16T21:32:46.157945Z</value>
          <lowerUncertainty>0.1000000015</lowerUncertainty>
          <upperUncertainty>0.1000000015</upperUncertainty>
        </time>
        <waveformID networkCode="TX" stationCode="PB24" locationCode="00" channelCode="HHZ"></waveformID>
        <phaseHint>P</phaseHint>
        <evaluationMode>manual</evaluationMode>
        <creationInfo>
          <agencyID>TXNet</agencyID>
          <author>daniel</author>
          <creationTime>2022-11-17T01:59:38.385218Z</creationTime>
        </creationInfo>
      </pick>
      <pick publicID="smi:org.gfz-potsdam.de/geofon/Pick/20221117015938.385263.74511">
        <time>
          <value>2022-11-16T21:32:46.256619Z</value>
          <lowerUncertainty>0.1000000015</lowerUncertainty>
          <upperUncertainty>0.1000000015</upperUncertainty>
        </time>
        <waveformID networkCode="4O" stationCode="WB03" locationCode="00" channelCode="HHN"></waveformID>
        <phaseHint>P</phaseHint>
        <evaluationMode>manual</evaluationMode>
        <creationInfo>
          <agencyID>TXNet</agencyID>
          <author>daniel</author>
          <creationTime>2022-11-17T01:59:38.385276Z</creationTime>
        </creationInfo>
      </pick>
      <pick publicID="smi:org.gfz-potsdam.de/geofon/Pick/20221117015938.3853.74512">
        <time>
          <value>2022-11-16T21:32:46.573785Z</value>
          <lowerUncertainty>0.1000000015</lowerUncertainty>
          <upperUncertainty>0.1000000015</upperUncertainty>
        </time>
        <waveformID networkCode="TX" stationCode="PB20" locationCode="00" channelCode="HHN"></waveformID>
        <phaseHint>P</phaseHint>
        <evaluationMode>manual</evaluationMode>
        <creationInfo>
          <agencyID>TXNet</agencyID>
          <author>daniel</author>
          <creationTime>2022-11-17T01:59:38.385314Z</creationTime>
        </creationInfo>
      </pick>
      <pick publicID="smi:org.gfz-potsdam.de/geofon/Pick/20221117015938.385627.74518">
        <time>
          <value>2022-11-16T21:32:48.307627Z</value>
          <lowerUncertainty>0.1000000015</lowerUncertainty>
          <upperUncertainty>0.1000000015</upperUncertainty>
        </time>
        <waveformID networkCode="TX" stationCode="PB23" locationCode="00" channelCode="HHE"></waveformID>
        <phaseHint>S</phaseHint>
        <evaluationMode>manual</evaluationMode>
        <creationInfo>
          <agencyID>TXNet</agencyID>
          <author>daniel</author>
          <creationTime>2022-11-17T01:59:38.385639Z</creationTime>
        </creationInfo>
      </pick>
      <pick publicID="smi:org.gfz-potsdam.de/geofon/Pick/20221117015938.385718.74520">
        <time>
          <value>2022-11-16T21:32:46.919144Z</value>
          <lowerUncertainty>0.1000000015</lowerUncertainty>
          <upperUncertainty>0.1000000015</upperUncertainty>
        </time>
        <waveformID networkCode="TX" stationCode="PB25" locationCode="00" channelCode="HHN"></waveformID>
        <phaseHint>P</phaseHint>
        <evaluationMode>manual</evaluationMode>
        <creationInfo>
          <agencyID>TXNet</agencyID>
          <author>daniel</author>
          <creationTime>2022-11-17T01:59:38.385731Z</creationTime>
        </creationInfo>
      </pick>
      <pick publicID="smi:org.gfz-potsdam.de/geofon/Pick/20221117015938.385851.74521">
        <time>
          <value>2022-11-16T21:32:47.349081Z</value>
          <lowerUncertainty>0.1000000015</lowerUncertainty>
          <upperUncertainty>0.1000000015</upperUncertainty>
        </time>
        <waveformID networkCode="TX" stationCode="PB31" locationCode="00" channelCode="HH1"></waveformID>
        <phaseHint>P</phaseHint>
        <evaluationMode>manual</evaluationMode>
        <creationInfo>
          <agencyID>TXNet</agencyID>
          <author>daniel</author>
          <creationTime>2022-11-17T01:59:38.385864Z</creationTime>
        </creationInfo>
      </pick>
      <pick publicID="smi:org.gfz-potsdam.de/geofon/Pick/20221117015938.385932.74523">
        <time>
          <value>2022-11-16T21:32:47.398627Z</value>
          <lowerUncertainty>0.1000000015</lowerUncertainty>
          <upperUncertainty>0.1000000015</upperUncertainty>
        </time>
        <waveformID networkCode="4O" stationCode="WB04" locationCode="00" channelCode="HHN"></waveformID>
        <phaseHint>P</phaseHint>
        <evaluationMode>manual</evaluationMode>
        <creationInfo>
          <agencyID>TXNet</agencyID>
          <author>daniel</author>
          <creationTime>2022-11-17T01:59:38.385945Z</creationTime>
        </creationInfo>
      </pick>
      <pick publicID="smi:org.gfz-potsdam.de/geofon/Pick/20221117015938.386056.74526">
        <time>
          <value>2022-11-16T21:32:47.517653Z</value>
          <lowerUncertainty>0.1000000015</lowerUncertainty>
          <upperUncertainty>0.1000000015</upperUncertainty>
        </time>
        <waveformID networkCode="4O" stationCode="WB02" locationCode="00" channelCode="HHN"></waveformID>
        <phaseHint>P</phaseHint>
        <evaluationMode>manual</evaluationMode>
        <creationInfo>
          <agencyID>TXNet</agencyID>
          <author>daniel</author>
          <creationTime>2022-11-17T01:59:38.386069Z</creationTime>
        </creationInfo>
      </pick>
      <pick publicID="smi:org.gfz-potsdam.de/geofon/Pick/20221117015938.38619.74529">
        <time>
          <value>2022-11-16T21:32:48.230098Z</value>
          <lowerUncertainty>0.1000000015</lowerUncertainty>
          <upperUncertainty>0.1000000015</upperUncertainty>
        </time>
        <waveformID networkCode="TX" stationCode="PB13" locationCode="01" channelCode="HHN"></waveformID>
        <phaseHint>P</phaseHint>
        <polarity>negative</polarity>
        <evaluationMode>manual</evaluationMode>
        <creationInfo>
          <agencyID>TXNet</agencyID>
          <author>daniel</author>
          <creationTime>2022-11-17T01:59:38.386204Z</creationTime>
        </creationInfo>
      </pick>
      <pick publicID="smi:org.gfz-potsdam.de/geofon/Pick/20221117015938.386355.74530">
        <time>
          <value>2022-11-16T21:32:49.498763Z</value>
          <lowerUncertainty>0.1000000015</lowerUncertainty>
          <upperUncertainty>0.1000000015</upperUncertainty>
        </time>
        <waveformID networkCode="4O" stationCode="CV01" channelCode="HHN"></waveformID>
        <phaseHint>P</phaseHint>
        <evaluationMode>manual</evaluationMode>
        <creationInfo>
          <agencyID>TXNet</agencyID>
          <author>daniel</author>
          <creationTime>2022-11-17T01:59:38.386387Z</creationTime>
        </creationInfo>
      </pick>
      <pick publicID="smi:org.gfz-potsdam.de/geofon/Pick/20221117015938.386418.74531">
        <time>
          <value>2022-11-16T21:32:53.107409Z</value>
          <lowerUncertainty>0.200000003</lowerUncertainty>
          <upperUncertainty>0.200000003</upperUncertainty>
        </time>
        <waveformID networkCode="4O" stationCode="CV01" channelCode="HHE"></waveformID>
        <phaseHint>S</phaseHint>
        <evaluationMode>manual</evaluationMode>
        <creationInfo>
          <agencyID>TXNet</agencyID>
          <author>daniel</author>
          <creationTime>2022-11-17T01:59:38.386431Z</creationTime>
        </creationInfo>
      </pick>
      <pick publicID="smi:org.gfz-potsdam.de/geofon/Pick/20221117015938.38662.74533">
        <time>
          <value>2022-11-16T21:32:50.647609Z</value>
          <lowerUncertainty>0.1000000015</lowerUncertainty>
          <upperUncertainty>0.1000000015</upperUncertainty>
        </time>
        <waveformID networkCode="4O" stationCode="CT01" channelCode="HHZ"></waveformID>
        <phaseHint>P</phaseHint>
        <polarity>positive</polarity>
        <evaluationMode>manual</evaluationMode>
        <creationInfo>
          <agencyID>TXNet</agencyID>
          <author>daniel</author>
          <creationTime>2022-11-17T01:59:38.386634Z</creationTime>
        </creationInfo>
      </pick>
      <pick publicID="smi:org.gfz-potsdam.de/geofon/Pick/20221117015938.386661.74534">
        <time>
          <value>2022-11-16T21:32:55.235947Z</value>
          <lowerUncertainty>0.3000000119</lowerUncertainty>
          <upperUncertainty>0.3000000119</upperUncertainty>
        </time>
        <waveformID networkCode="4O" stationCode="CT01" channelCode="HHE"></waveformID>
        <phaseHint>S</phaseHint>
        <evaluationMode>manual</evaluationMode>
        <creationInfo>
          <agencyID>TXNet</agencyID>
          <author>daniel</author>
          <creationTime>2022-11-17T01:59:38.386677Z</creationTime>
        </creationInfo>
      </pick>
      <pick publicID="smi:org.gfz-potsdam.de/geofon/Pick/20221117015938.386699.74535">
        <time>
          <value>2022-11-16T21:32:50.746283Z</value>
          <lowerUncertainty>0.1000000015</lowerUncertainty>
          <upperUncertainty>0.1000000015</upperUncertainty>
        </time>
        <waveformID networkCode="TX" stationCode="PB11" locationCode="00" channelCode="HH1"></waveformID>
        <phaseHint>P</phaseHint>
        <polarity>positive</polarity>
        <evaluationMode>manual</evaluationMode>
        <creationInfo>
          <agencyID>TXNet</agencyID>
          <author>daniel</author>
          <creationTime>2022-11-17T01:59:38.386713Z</creationTime>
        </creationInfo>
      </pick>
      <pick publicID="smi:org.gfz-potsdam.de/geofon/Pick/20221117015938.386736.74536">
        <time>
          <value>2022-11-16T21:32:55.771606Z</value>
          <lowerUncertainty>0.200000003</lowerUncertainty>
          <upperUncertainty>0.200000003</upperUncertainty>
        </time>
        <waveformID networkCode="TX" stationCode="PB11" locationCode="00" channelCode="HH2"></waveformID>
        <phaseHint>S</phaseHint>
        <evaluationMode>manual</evaluationMode>
        <creationInfo>
          <agencyID>TXNet</agencyID>
          <author>daniel</author>
          <creationTime>2022-11-17T01:59:38.386768Z</creationTime>
        </creationInfo>
      </pick>
      <pick publicID="smi:org.gfz-potsdam.de/geofon/Pick/20221117015938.386792.74537">
        <time>
          <value>2022-11-16T21:32:51.437001Z</value>
          <lowerUncertainty>0.1000000015</lowerUncertainty>
          <upperUncertainty>0.1000000015</upperUncertainty>
        </time>
        <waveformID networkCode="4O" stationCode="WB12" locationCode="00" channelCode="HHN"></waveformID>
        <phaseHint>P</phaseHint>
        <polarity>positive</polarity>
        <evaluationMode>manual</evaluationMode>
        <creationInfo>
          <agencyID>TXNet</agencyID>
          <author>daniel</author>
          <creationTime>2022-11-17T01:59:38.386805Z</creationTime>
        </creationInfo>
      </pick>
      <pick publicID="smi:org.gfz-potsdam.de/geofon/Pick/20221117015938.386847.74538">
        <time>
          <value>2022-11-16T21:32:57.420870Z</value>
          <lowerUncertainty>0.200000003</lowerUncertainty>
          <upperUncertainty>0.200000003</upperUncertainty>
        </time>
        <waveformID networkCode="4O" stationCode="WB12" locationCode="00" channelCode="HHN"></waveformID>
        <phaseHint>S</phaseHint>
        <evaluationMode>manual</evaluationMode>
        <creationInfo>
          <agencyID>TXNet</agencyID>
          <author>daniel</author>
          <creationTime>2022-11-17T01:59:38.386863Z</creationTime>
        </creationInfo>
      </pick>
      <pick publicID="smi:org.gfz-potsdam.de/geofon/Pick/20221117015938.386886.74539">
        <time>
          <value>2022-11-16T21:32:51.606156Z</value>
          <lowerUncertainty>0.1000000015</lowerUncertainty>
          <upperUncertainty>0.1000000015</upperUncertainty>
        </time>
        <waveformID networkCode="TX" stationCode="PB42" locationCode="00" channelCode="HHN"></waveformID>
        <phaseHint>P</phaseHint>
        <polarity>positive</polarity>
        <evaluationMode>manual</evaluationMode>
        <creationInfo>
          <agencyID>TXNet</agencyID>
          <author>daniel</author>
          <creationTime>2022-11-17T01:59:38.386899Z</creationTime>
        </creationInfo>
      </pick>
      <pick publicID="smi:org.gfz-potsdam.de/geofon/Pick/20221117015938.386917.74540">
        <time>
          <value>2022-11-16T21:32:51.366519Z</value>
          <lowerUncertainty>0.1000000015</lowerUncertainty>
          <upperUncertainty>0.1000000015</upperUncertainty>
        </time>
        <waveformID networkCode="4O" stationCode="CT02" channelCode="HHN"></waveformID>
        <phaseHint>P</phaseHint>
        <polarity>positive</polarity>
        <evaluationMode>manual</evaluationMode>
        <creationInfo>
          <agencyID>TXNet</agencyID>
          <author>daniel</author>
          <creationTime>2022-11-17T01:59:38.386933Z</creationTime>
        </creationInfo>
      </pick>
      <pick publicID="smi:org.gfz-potsdam.de/geofon/Pick/20221117015938.386951.74541">
        <time>
          <value>2022-11-16T21:32:56.687864Z</value>
          <lowerUncertainty>0.200000003</lowerUncertainty>
          <upperUncertainty>0.200000003</upperUncertainty>
        </time>
        <waveformID networkCode="4O" stationCode="CT02" channelCode="HHN"></waveformID>
        <phaseHint>S</phaseHint>
        <evaluationMode>manual</evaluationMode>
        <creationInfo>
          <agencyID>TXNet</agencyID>
          <author>daniel</author>
          <creationTime>2022-11-17T01:59:38.386966Z</creationTime>
        </creationInfo>
      </pick>
      <pick publicID="smi:org.gfz-potsdam.de/geofon/Pick/20221117015938.386988.74542">
        <time>
          <value>2022-11-16T21:32:51.584855Z</value>
          <lowerUncertainty>0.1000000015</lowerUncertainty>
          <upperUncertainty>0.1000000015</upperUncertainty>
        </time>
        <waveformID networkCode="4O" stationCode="WB11" locationCode="00" channelCode="HHE"></waveformID>
        <phaseHint>P</phaseHint>
        <evaluationMode>manual</evaluationMode>
        <creationInfo>
          <agencyID>TXNet</agencyID>
          <author>daniel</author>
          <creationTime>2022-11-17T01:59:38.387010Z</creationTime>
        </creationInfo>
      </pick>
      <pick publicID="smi:org.gfz-potsdam.de/geofon/Pick/20221117015938.387074.74544">
        <time>
          <value>2022-11-16T21:32:52.027869Z</value>
        </time>
        <waveformID networkCode="TX" stationCode="PB35" locationCode="00" channelCode="HH2"></waveformID>
        <phaseHint>P</phaseHint>
        <evaluationMode>manual</evaluationMode>
        <creationInfo>
          <agencyID>TXNet</agencyID>
          <author>daniel</author>
          <creationTime>2022-11-17T01:59:38.387085Z</creationTime>
        </creationInfo>
      </pick>
      <pick publicID="smi:org.gfz-potsdam.de/geofon/Pick/20221117015938.387093.74545">
        <time>
          <value>2022-11-16T21:32:52.162924Z</value>
          <lowerUncertainty>0.1000000015</lowerUncertainty>
          <upperUncertainty>0.1000000015</upperUncertainty>
        </time>
        <waveformID networkCode="TX" stationCode="PB38" locationCode="00" channelCode="HH2"></waveformID>
        <phaseHint>P</phaseHint>
        <evaluationMode>manual</evaluationMode>
        <creationInfo>
          <agencyID>TXNet</agencyID>
          <author>daniel</author>
          <creationTime>2022-11-17T01:59:38.387102Z</creationTime>
        </creationInfo>
      </pick>
      <pick publicID="smi:org.gfz-potsdam.de/geofon/Pick/20221117015938.387124.74546">
        <time>
          <value>2022-11-16T21:32:52.195018Z</value>
          <lowerUncertainty>0.1000000015</lowerUncertainty>
          <upperUncertainty>0.1000000015</upperUncertainty>
        </time>
        <waveformID networkCode="4O" stationCode="WB09" locationCode="00" channelCode="HHE"></waveformID>
        <phaseHint>P</phaseHint>
        <polarity>positive</polarity>
        <evaluationMode>manual</evaluationMode>
        <creationInfo>
          <agencyID>TXNet</agencyID>
          <author>daniel</author>
          <creationTime>2022-11-17T01:59:38.387138Z</creationTime>
        </creationInfo>
      </pick>
      <pick publicID="smi:org.gfz-potsdam.de/geofon/Pick/20221117015938.387156.74547">
        <time>
          <value>2022-11-16T21:32:52.261760Z</value>
          <lowerUncertainty>0.1000000015</lowerUncertainty>
          <upperUncertainty>0.1000000015</upperUncertainty>
        </time>
        <waveformID networkCode="GM" stationCode="NMP11" locationCode="01" channelCode="HHE"></waveformID>
        <phaseHint>P</phaseHint>
        <polarity>positive</polarity>
        <evaluationMode>manual</evaluationMode>
        <creationInfo>
          <agencyID>TXNet</agencyID>
          <author>daniel</author>
          <creationTime>2022-11-17T01:59:38.387172Z</creationTime>
        </creationInfo>
      </pick>
      <pick publicID="smi:org.gfz-potsdam.de/geofon/Pick/20221117015938.387195.74548">
        <time>
          <value>2022-11-16T21:32:52.230648Z</value>
          <lowerUncertainty>0.1000000015</lowerUncertainty>
          <upperUncertainty>0.1000000015</upperUncertainty>
        </time>
        <waveformID networkCode="4O" stationCode="SA07" locationCode="00" channelCode="HHE"></waveformID>
        <phaseHint>P</phaseHint>
        <evaluationMode>manual</evaluationMode>
        <creationInfo>
          <agencyID>TXNet</agencyID>
          <author>daniel</author>
          <creationTime>2022-11-17T01:59:38.387211Z</creationTime>
        </creationInfo>
      </pick>
      <pick publicID="smi:org.gfz-potsdam.de/geofon/Pick/20221117015938.387232.74549">
        <time>
          <value>2022-11-16T21:32:52.367785Z</value>
          <lowerUncertainty>0.1000000015</lowerUncertainty>
          <upperUncertainty>0.1000000015</upperUncertainty>
        </time>
        <waveformID networkCode="4O" stationCode="WB10" locationCode="00" channelCode="HHE"></waveformID>
        <phaseHint>P</phaseHint>
        <polarity>positive</polarity>
        <evaluationMode>manual</evaluationMode>
        <creationInfo>
          <agencyID>TXNet</agencyID>
          <author>daniel</author>
          <creationTime>2022-11-17T01:59:38.387246Z</creationTime>
        </creationInfo>
      </pick>
      <pick publicID="smi:org.gfz-potsdam.de/geofon/Pick/20221117015938.387366.74550">
        <time>
          <value>2022-11-16T21:32:53.330892Z</value>
          <lowerUncertainty>0.1000000015</lowerUncertainty>
          <upperUncertainty>0.1000000015</upperUncertainty>
        </time>
        <waveformID networkCode="4O" stationCode="WB06" locationCode="00" channelCode="HHE"></waveformID>
        <phaseHint>P</phaseHint>
        <polarity>positive</polarity>
        <evaluationMode>manual</evaluationMode>
        <creationInfo>
          <agencyID>TXNet</agencyID>
          <author>daniel</author>
          <creationTime>2022-11-17T01:59:38.387379Z</creationTime>
        </creationInfo>
      </pick>
      <pick publicID="smi:org.gfz-potsdam.de/geofon/Pick/20221117015938.387442.74551">
        <time>
          <value>2022-11-16T21:32:53.422649Z</value>
          <lowerUncertainty>0.1000000015</lowerUncertainty>
          <upperUncertainty>0.1000000015</upperUncertainty>
        </time>
        <waveformID networkCode="TX" stationCode="PB29" locationCode="00" channelCode="HH2"></waveformID>
        <phaseHint>P</phaseHint>
        <evaluationMode>manual</evaluationMode>
        <creationInfo>
          <agencyID>TXNet</agencyID>
          <author>daniel</author>
          <creationTime>2022-11-17T01:59:38.387467Z</creationTime>
        </creationInfo>
      </pick>
      <pick publicID="smi:org.gfz-potsdam.de/geofon/Pick/20221117015938.387489.74552">
        <time>
          <value>2022-11-16T21:32:53.607442Z</value>
          <lowerUncertainty>0.1000000015</lowerUncertainty>
          <upperUncertainty>0.1000000015</upperUncertainty>
        </time>
        <waveformID networkCode="TX" stationCode="PB37" locationCode="00" channelCode="HH2"></waveformID>
        <phaseHint>P</phaseHint>
        <evaluationMode>manual</evaluationMode>
        <creationInfo>
          <agencyID>TXNet</agencyID>
          <author>daniel</author>
          <creationTime>2022-11-17T01:59:38.387503Z</creationTime>
        </creationInfo>
      </pick>
      <pick publicID="smi:org.gfz-potsdam.de/geofon/Pick/20221117015938.387526.74553">
        <time>
          <value>2022-11-16T21:32:53.826723Z</value>
        </time>
        <waveformID networkCode="TX" stationCode="DG02" locationCode="00" channelCode="CH2"></waveformID>
        <phaseHint>P</phaseHint>
        <evaluationMode>manual</evaluationMode>
        <creationInfo>
          <agencyID>TXNet</agencyID>
          <author>daniel</author>
          <creationTime>2022-11-17T01:59:38.387539Z</creationTime>
        </creationInfo>
      </pick>
      <pick publicID="smi:org.gfz-potsdam.de/geofon/Pick/20221117015938.387596.74554">
        <time>
          <value>2022-11-16T21:32:53.260090Z</value>
          <lowerUncertainty>0.1000000015</lowerUncertainty>
          <upperUncertainty>0.1000000015</upperUncertainty>
        </time>
        <waveformID networkCode="4O" stationCode="SA09" locationCode="00" channelCode="HHN"></waveformID>
        <phaseHint>P</phaseHint>
        <evaluationMode>manual</evaluationMode>
        <creationInfo>
          <agencyID>TXNet</agencyID>
          <author>daniel</author>
          <creationTime>2022-11-17T01:59:38.387612Z</creationTime>
        </creationInfo>
      </pick>
      <pick publicID="smi:org.gfz-potsdam.de/geofon/Pick/20221117015938.387635.74555">
        <time>
          <value>2022-11-16T21:32:53.817462Z</value>
          <lowerUncertainty>0.1000000015</lowerUncertainty>
          <upperUncertainty>0.1000000015</upperUncertainty>
        </time>
        <waveformID networkCode="GM" stationCode="NMP31" locationCode="01" channelCode="HHN"></waveformID>
        <phaseHint>P</phaseHint>
        <evaluationMode>manual</evaluationMode>
        <creationInfo>
          <agencyID>TXNet</agencyID>
          <author>daniel</author>
          <creationTime>2022-11-17T01:59:38.387648Z</creationTime>
        </creationInfo>
      </pick>
      <pick publicID="smi:org.gfz-potsdam.de/geofon/Pick/20221117015938.387729.74556">
        <time>
          <value>2022-11-16T21:32:53.965989Z</value>
          <lowerUncertainty>0.1000000015</lowerUncertainty>
          <upperUncertainty>0.1000000015</upperUncertainty>
        </time>
        <waveformID networkCode="GM" stationCode="NMP02" locationCode="01" channelCode="HHN"></waveformID>
        <phaseHint>P</phaseHint>
        <evaluationMode>manual</evaluationMode>
        <creationInfo>
          <agencyID>TXNet</agencyID>
          <author>daniel</author>
          <creationTime>2022-11-17T01:59:38.387742Z</creationTime>
        </creationInfo>
      </pick>
      <pick publicID="smi:org.gfz-potsdam.de/geofon/Pick/20221117015938.387764.74557">
        <time>
          <value>2022-11-16T21:32:54.530689Z</value>
        </time>
        <waveformID networkCode="TX" stationCode="DG09" locationCode="00" channelCode="CH2"></waveformID>
        <phaseHint>P</phaseHint>
        <evaluationMode>manual</evaluationMode>
        <creationInfo>
          <agencyID>TXNet</agencyID>
          <author>daniel</author>
          <creationTime>2022-11-17T01:59:38.387778Z</creationTime>
        </creationInfo>
      </pick>
      <pick publicID="smi:org.gfz-potsdam.de/geofon/Pick/20221117015938.387801.74558">
        <time>
          <value>2022-11-16T21:32:54.606151Z</value>
        </time>
        <waveformID networkCode="TX" stationCode="DG06" locationCode="00" channelCode="CH2"></waveformID>
        <phaseHint>P</phaseHint>
        <evaluationMode>manual</evaluationMode>
        <creationInfo>
          <agencyID>TXNet</agencyID>
          <author>daniel</author>
          <creationTime>2022-11-17T01:59:38.387815Z</creationTime>
        </creationInfo>
      </pick>
      <pick publicID="smi:org.gfz-potsdam.de/geofon/Pick/20221117015938.387838.74559">
        <time>
          <value>2022-11-16T21:32:54.283347Z</value>
          <lowerUncertainty>0.1000000015</lowerUncertainty>
          <upperUncertainty>0.1000000015</upperUncertainty>
        </time>
        <waveformID networkCode="TX" stationCode="DG08" locationCode="00" channelCode="CH1"></waveformID>
        <phaseHint>P</phaseHint>
        <evaluationMode>manual</evaluationMode>
        <creationInfo>
          <agencyID>TXNet</agencyID>
          <author>daniel</author>
          <creationTime>2022-11-17T01:59:38.387852Z</creationTime>
        </creationInfo>
      </pick>
      <pick publicID="smi:org.gfz-potsdam.de/geofon/Pick/20221117015938.387875.74560">
        <time>
          <value>2022-11-16T21:32:54.335667Z</value>
          <lowerUncertainty>0.1000000015</lowerUncertainty>
          <upperUncertainty>0.1000000015</upperUncertainty>
        </time>
        <waveformID networkCode="GM" stationCode="NMP12" locationCode="01" channelCode="HHN"></waveformID>
        <phaseHint>P</phaseHint>
        <polarity>positive</polarity>
        <evaluationMode>manual</evaluationMode>
        <creationInfo>
          <agencyID>TXNet</agencyID>
          <author>daniel</author>
          <creationTime>2022-11-17T01:59:38.387890Z</creationTime>
        </creationInfo>
      </pick>
      <pick publicID="smi:org.gfz-potsdam.de/geofon/Pick/20221117015938.387909.74561">
        <time>
          <value>2022-11-16T21:32:54.794496Z</value>
        </time>
        <waveformID networkCode="TX" stationCode="DG03" locationCode="00" channelCode="CH2"></waveformID>
        <phaseHint>P</phaseHint>
        <evaluationMode>manual</evaluationMode>
        <creationInfo>
          <agencyID>TXNet</agencyID>
          <author>daniel</author>
          <creationTime>2022-11-17T01:59:38.387924Z</creationTime>
        </creationInfo>
      </pick>
      <pick publicID="smi:org.gfz-potsdam.de/geofon/Pick/20221117015938.388009.74562">
        <time>
          <value>2022-11-16T21:32:54.500765Z</value>
          <lowerUncertainty>0.1000000015</lowerUncertainty>
          <upperUncertainty>0.1000000015</upperUncertainty>
        </time>
        <waveformID networkCode="GM" stationCode="NMP41" locationCode="01" channelCode="HHN"></waveformID>
        <phaseHint>P</phaseHint>
        <evaluationMode>manual</evaluationMode>
        <creationInfo>
          <agencyID>TXNet</agencyID>
          <author>daniel</author>
          <creationTime>2022-11-17T01:59:38.388023Z</creationTime>
        </creationInfo>
      </pick>
      <pick publicID="smi:org.gfz-potsdam.de/geofon/Pick/20221117020158.518175.74663">
        <time>
          <value>2022-11-16T21:32:56.909280Z</value>
        </time>
        <waveformID networkCode="TX" stationCode="PB42" locationCode="00" channelCode="HHE"></waveformID>
        <phaseHint>S</phaseHint>
        <evaluationMode>manual</evaluationMode>
        <creationInfo>
          <agencyID>TXNet</agencyID>
          <author>daniel</author>
          <creationTime>2022-11-17T02:01:58.518206Z</creationTime>
        </creationInfo>
      </pick>
      <pick publicID="smi:org.gfz-potsdam.de/geofon/Pick/20221117020158.518298.74664">
        <time>
          <value>2022-11-16T21:32:58.706927Z</value>
        </time>
        <waveformID networkCode="4O" stationCode="WB11" locationCode="00" channelCode="HHE"></waveformID>
        <phaseHint>S</phaseHint>
        <evaluationMode>manual</evaluationMode>
        <creationInfo>
          <agencyID>TXNet</agencyID>
          <author>daniel</author>
          <creationTime>2022-11-17T02:01:58.518312Z</creationTime>
        </creationInfo>
      </pick>
      <pick publicID="smi:org.gfz-potsdam.de/geofon/Pick/20221117020158.51843.74666">
        <time>
          <value>2022-11-16T21:32:56.509409Z</value>
        </time>
        <waveformID networkCode="TX" stationCode="PB35" locationCode="00" channelCode="HH1"></waveformID>
        <phaseHint>S</phaseHint>
        <evaluationMode>manual</evaluationMode>
        <creationInfo>
          <agencyID>TXNet</agencyID>
          <author>daniel</author>
          <creationTime>2022-11-17T02:01:58.518444Z</creationTime>
        </creationInfo>
      </pick>
      <pick publicID="smi:org.gfz-potsdam.de/geofon/Pick/20221117020158.51849.74667">
        <time>
          <value>2022-11-16T21:32:57.236992Z</value>
        </time>
        <waveformID networkCode="TX" stationCode="PB38" locationCode="00" channelCode="HH1"></waveformID>
        <phaseHint>S</phaseHint>
        <evaluationMode>manual</evaluationMode>
        <creationInfo>
          <agencyID>TXNet</agencyID>
          <author>daniel</author>
          <creationTime>2022-11-17T02:01:58.518503Z</creationTime>
        </creationInfo>
      </pick>
      <pick publicID="smi:org.gfz-potsdam.de/geofon/Pick/20221117020158.518547.74668">
        <time>
          <value>2022-11-16T21:32:59.785279Z</value>
        </time>
        <waveformID networkCode="4O" stationCode="WB09" locationCode="00" channelCode="HHE"></waveformID>
        <phaseHint>S</phaseHint>
        <evaluationMode>manual</evaluationMode>
        <creationInfo>
          <agencyID>TXNet</agencyID>
          <author>daniel</author>
          <creationTime>2022-11-17T02:01:58.518561Z</creationTime>
        </creationInfo>
      </pick>
      <pick publicID="smi:org.gfz-potsdam.de/geofon/Pick/20221117020158.518622.74669">
        <time>
          <value>2022-11-16T21:32:58.218362Z</value>
        </time>
        <waveformID networkCode="GM" stationCode="NMP11" locationCode="01" channelCode="HHE"></waveformID>
        <phaseHint>S</phaseHint>
        <evaluationMode>manual</evaluationMode>
        <creationInfo>
          <agencyID>TXNet</agencyID>
          <author>daniel</author>
          <creationTime>2022-11-17T02:01:58.518636Z</creationTime>
        </creationInfo>
      </pick>
      <pick publicID="smi:org.gfz-potsdam.de/geofon/Pick/20221117020158.518706.74670">
        <time>
          <value>2022-11-16T21:32:57.389218Z</value>
        </time>
        <waveformID networkCode="4O" stationCode="SA07" locationCode="00" channelCode="HHE"></waveformID>
        <phaseHint>S</phaseHint>
        <evaluationMode>manual</evaluationMode>
        <creationInfo>
          <agencyID>TXNet</agencyID>
          <author>daniel</author>
          <creationTime>2022-11-17T02:01:58.518721Z</creationTime>
        </creationInfo>
      </pick>
      <pick publicID="smi:org.gfz-potsdam.de/geofon/Pick/20221117020158.518822.74671">
        <time>
          <value>2022-11-16T21:32:59.161213Z</value>
        </time>
        <waveformID networkCode="4O" stationCode="WB10" locationCode="00" channelCode="HHN"></waveformID>
        <phaseHint>S</phaseHint>
        <evaluationMode>manual</evaluationMode>
        <creationInfo>
          <agencyID>TXNet</agencyID>
          <author>daniel</author>
          <creationTime>2022-11-17T02:01:58.518836Z</creationTime>
        </creationInfo>
      </pick>
      <pick publicID="smi:org.gfz-potsdam.de/geofon/Pick/20221117020158.518935.74672">
        <time>
          <value>2022-11-16T21:32:58.733415Z</value>
        </time>
        <waveformID networkCode="TX" stationCode="PB12" locationCode="00" channelCode="HHE"></waveformID>
        <phaseHint>S</phaseHint>
        <evaluationMode>manual</evaluationMode>
        <creationInfo>
          <agencyID>TXNet</agencyID>
          <author>daniel</author>
          <creationTime>2022-11-17T02:01:58.518948Z</creationTime>
        </creationInfo>
      </pick>
      <pick publicID="smi:org.gfz-potsdam.de/geofon/Pick/20221117020158.519003.74673">
        <time>
          <value>2022-11-16T21:32:59.026115Z</value>
        </time>
        <waveformID networkCode="TX" stationCode="PB28" locationCode="00" channelCode="HH2"></waveformID>
        <phaseHint>S</phaseHint>
        <evaluationMode>manual</evaluationMode>
        <creationInfo>
          <agencyID>TXNet</agencyID>
          <author>daniel</author>
          <creationTime>2022-11-17T02:01:58.519018Z</creationTime>
        </creationInfo>
      </pick>
      <pick publicID="smi:org.gfz-potsdam.de/geofon/Pick/20221117020158.51915.74674">
        <time>
          <value>2022-11-16T21:32:59.905770Z</value>
        </time>
        <waveformID networkCode="TX" stationCode="PB37" locationCode="00" channelCode="HH1"></waveformID>
        <phaseHint>S</phaseHint>
        <evaluationMode>manual</evaluationMode>
        <creationInfo>
          <agencyID>TXNet</agencyID>
          <author>daniel</author>
          <creationTime>2022-11-17T02:01:58.519163Z</creationTime>
        </creationInfo>
      </pick>
      <pick publicID="smi:org.gfz-potsdam.de/geofon/Pick/20221117020158.519218.74675">
        <time>
          <value>2022-11-16T21:33:00.268333Z</value>
        </time>
        <waveformID networkCode="TX" stationCode="DG02" locationCode="00" channelCode="CH1"></waveformID>
        <phaseHint>S</phaseHint>
        <evaluationMode>manual</evaluationMode>
        <creationInfo>
          <agencyID>TXNet</agencyID>
          <author>daniel</author>
          <creationTime>2022-11-17T02:01:58.519232Z</creationTime>
        </creationInfo>
      </pick>
      <pick publicID="smi:org.gfz-potsdam.de/geofon/Pick/20221117020158.519584.74677">
        <time>
          <value>2022-11-16T21:32:55.932550Z</value>
        </time>
        <waveformID networkCode="GM" stationCode="NMP01" locationCode="01" channelCode="HHN"></waveformID>
        <phaseHint>P</phaseHint>
        <evaluationMode>manual</evaluationMode>
        <creationInfo>
          <agencyID>TXNet</agencyID>
          <author>daniel</author>
          <creationTime>2022-11-17T02:01:58.519597Z</creationTime>
        </creationInfo>
      </pick>
      <pick publicID="smi:org.gfz-potsdam.de/geofon/Pick/20221117020158.519658.74679">
        <time>
          <value>2022-11-16T21:32:56.530007Z</value>
          <lowerUncertainty>0.1000000015</lowerUncertainty>
          <upperUncertainty>0.1000000015</upperUncertainty>
        </time>
        <waveformID networkCode="GM" stationCode="NMP51" locationCode="01" channelCode="HHN"></waveformID>
        <phaseHint>P</phaseHint>
        <evaluationMode>manual</evaluationMode>
        <creationInfo>
          <agencyID>TXNet</agencyID>
          <author>daniel</author>
          <creationTime>2022-11-17T02:01:58.519672Z</creationTime>
        </creationInfo>
      </pick>
      <pick publicID="smi:org.gfz-potsdam.de/geofon/Pick/20221117020158.519696.74680">
        <time>
          <value>2022-11-16T21:32:56.992849Z</value>
        </time>
        <waveformID networkCode="SC" stationCode="CL2B" locationCode="00" channelCode="EH?"></waveformID>
        <phaseHint>P</phaseHint>
        <evaluationMode>manual</evaluationMode>
        <creationInfo>
          <agencyID>TXNet</agencyID>
          <author>daniel</author>
          <creationTime>2022-11-17T02:01:58.519711Z</creationTime>
        </creationInfo>
      </pick>
      <pick publicID="smi:org.gfz-potsdam.de/geofon/Pick/20221117020158.519734.74681">
        <time>
          <value>2022-11-16T21:32:56.698762Z</value>
        </time>
        <waveformID networkCode="SC" stationCode="GDL2" locationCode="00" channelCode="HHN"></waveformID>
        <phaseHint>P</phaseHint>
        <evaluationMode>manual</evaluationMode>
        <creationInfo>
          <agencyID>TXNet</agencyID>
          <author>daniel</author>
          <creationTime>2022-11-17T02:01:58.519747Z</creationTime>
        </creationInfo>
      </pick>
      <pick publicID="smi:org.gfz-potsdam.de/geofon/Pick/20221117020158.51977.74682">
        <time>
          <value>2022-11-16T21:32:57.203028Z</value>
        </time>
        <waveformID networkCode="4O" stationCode="PL01" locationCode="00" channelCode="HHN"></waveformID>
        <phaseHint>P</phaseHint>
        <evaluationMode>manual</evaluationMode>
        <creationInfo>
          <agencyID>TXNet</agencyID>
          <author>daniel</author>
          <creationTime>2022-11-17T02:01:58.519785Z</creationTime>
        </creationInfo>
      </pick>
      <pick publicID="smi:org.gfz-potsdam.de/geofon/Pick/20221117020158.519808.74683">
        <time>
          <value>2022-11-16T21:32:57.235589Z</value>
        </time>
        <waveformID networkCode="GM" stationCode="NMP44" locationCode="01" channelCode="HHN"></waveformID>
        <phaseHint>P</phaseHint>
        <evaluationMode>manual</evaluationMode>
        <creationInfo>
          <agencyID>TXNet</agencyID>
          <author>daniel</author>
          <creationTime>2022-11-17T02:01:58.519822Z</creationTime>
        </creationInfo>
      </pick>
      <pick publicID="smi:org.gfz-potsdam.de/geofon/Pick/20221117164826.532236.115001">
        <time>
          <value>2022-11-16T21:32:46.536693Z</value>
          <lowerUncertainty>0.1000000015</lowerUncertainty>
          <upperUncertainty>0.1000000015</upperUncertainty>
        </time>
        <waveformID networkCode="TX" stationCode="PB40" locationCode="00" channelCode="HNZ"></waveformID>
        <phaseHint>P</phaseHint>
        <polarity>negative</polarity>
        <evaluationMode>manual</evaluationMode>
        <creationInfo>
          <agencyID>TXNet</agencyID>
          <author>daniel</author>
          <creationTime>2022-11-17T16:48:26.532298Z</creationTime>
        </creationInfo>
      </pick>
      <pick publicID="smi:org.gfz-potsdam.de/geofon/Pick/20221117164826.5324.115003">
        <time>
          <value>2022-11-16T21:32:46.806374Z</value>
          <lowerUncertainty>0.1000000015</lowerUncertainty>
          <upperUncertainty>0.1000000015</upperUncertainty>
        </time>
        <waveformID networkCode="TX" stationCode="PB23" locationCode="00" channelCode="HHN"></waveformID>
        <phaseHint>P</phaseHint>
        <polarity>positive</polarity>
        <evaluationMode>manual</evaluationMode>
        <creationInfo>
          <agencyID>TXNet</agencyID>
          <author>daniel</author>
          <creationTime>2022-11-17T16:48:26.532423Z</creationTime>
        </creationInfo>
      </pick>
      <pick publicID="smi:org.gfz-potsdam.de/geofon/Pick/20221117164826.53251.115004">
        <time>
          <value>2022-11-16T21:32:47.395833Z</value>
          <lowerUncertainty>0.1000000015</lowerUncertainty>
          <upperUncertainty>0.1000000015</upperUncertainty>
        </time>
        <waveformID networkCode="TX" stationCode="PB33" locationCode="00" channelCode="HNZ"></waveformID>
        <phaseHint>P</phaseHint>
        <evaluationMode>manual</evaluationMode>
        <creationInfo>
          <agencyID>TXNet</agencyID>
          <author>daniel</author>
          <creationTime>2022-11-17T16:48:26.532533Z</creationTime>
        </creationInfo>
      </pick>
      <pick publicID="smi:org.gfz-potsdam.de/geofon/Pick/20221117164826.532567.115005">
        <time>
          <value>2022-11-16T21:32:49.778777Z</value>
          <lowerUncertainty>0.1000000015</lowerUncertainty>
          <upperUncertainty>0.1000000015</upperUncertainty>
        </time>
        <waveformID networkCode="TX" stationCode="PB33" locationCode="00" channelCode="HN2"></waveformID>
        <phaseHint>S</phaseHint>
        <evaluationMode>manual</evaluationMode>
        <creationInfo>
          <agencyID>TXNet</agencyID>
          <author>daniel</author>
          <creationTime>2022-11-17T16:48:26.532580Z</creationTime>
        </creationInfo>
      </pick>
      <pick publicID="smi:org.gfz-potsdam.de/geofon/Pick/20221117164826.532752.115006">
        <time>
          <value>2022-11-16T21:32:47.684791Z</value>
          <lowerUncertainty>0.1000000015</lowerUncertainty>
          <upperUncertainty>0.1000000015</upperUncertainty>
        </time>
        <waveformID networkCode="TX" stationCode="PB43" locationCode="00" channelCode="HNZ"></waveformID>
        <phaseHint>P</phaseHint>
        <polarity>positive</polarity>
        <evaluationMode>manual</evaluationMode>
        <creationInfo>
          <agencyID>TXNet</agencyID>
          <author>daniel</author>
          <creationTime>2022-11-17T16:48:26.532782Z</creationTime>
        </creationInfo>
      </pick>
      <pick publicID="smi:org.gfz-potsdam.de/geofon/Pick/20221117164826.532803.115007">
        <time>
          <value>2022-11-16T21:32:50.586002Z</value>
          <lowerUncertainty>0.200000003</lowerUncertainty>
          <upperUncertainty>0.200000003</upperUncertainty>
        </time>
        <waveformID networkCode="TX" stationCode="PB43" locationCode="00" channelCode="HN2"></waveformID>
        <phaseHint>S</phaseHint>
        <evaluationMode>manual</evaluationMode>
        <creationInfo>
          <agencyID>TXNet</agencyID>
          <author>daniel</author>
          <creationTime>2022-11-17T16:48:26.532822Z</creationTime>
        </creationInfo>
      </pick>
      <pick publicID="smi:org.gfz-potsdam.de/geofon/Pick/20221117164826.532898.115008">
        <time>
          <value>2022-11-16T21:32:51.057011Z</value>
          <lowerUncertainty>0.200000003</lowerUncertainty>
          <upperUncertainty>0.200000003</upperUncertainty>
        </time>
        <waveformID networkCode="TX" stationCode="PB13" locationCode="01" channelCode="HHN"></waveformID>
        <phaseHint>S</phaseHint>
        <evaluationMode>manual</evaluationMode>
        <creationInfo>
          <agencyID>TXNet</agencyID>
          <author>daniel</author>
          <creationTime>2022-11-17T16:48:26.532918Z</creationTime>
        </creationInfo>
      </pick>
      <pick publicID="smi:org.gfz-potsdam.de/geofon/Pick/20221117164826.533235.115009">
        <time>
          <value>2022-11-16T21:32:50.399685Z</value>
          <lowerUncertainty>0.1000000015</lowerUncertainty>
          <upperUncertainty>0.1000000015</upperUncertainty>
        </time>
        <waveformID networkCode="TX" stationCode="PECS" locationCode="00" channelCode="HN1"></waveformID>
        <phaseHint>P</phaseHint>
        <evaluationMode>manual</evaluationMode>
        <creationInfo>
          <agencyID>TXNet</agencyID>
          <author>daniel</author>
          <creationTime>2022-11-17T16:48:26.533272Z</creationTime>
        </creationInfo>
      </pick>
      <pick publicID="smi:org.gfz-potsdam.de/geofon/Pick/20221117164826.533295.115010">
        <time>
          <value>2022-11-16T21:32:54.676552Z</value>
          <lowerUncertainty>0.200000003</lowerUncertainty>
          <upperUncertainty>0.200000003</upperUncertainty>
        </time>
        <waveformID networkCode="TX" stationCode="PECS" locationCode="00" channelCode="HN1"></waveformID>
        <phaseHint>S</phaseHint>
        <evaluationMode>manual</evaluationMode>
        <creationInfo>
          <agencyID>TXNet</agencyID>
          <author>daniel</author>
          <creationTime>2022-11-17T16:48:26.533307Z</creationTime>
        </creationInfo>
      </pick>
      <pick publicID="smi:org.gfz-potsdam.de/geofon/Pick/20221117164826.533321.115011">
        <time>
          <value>2022-11-16T21:32:55.541597Z</value>
        </time>
        <waveformID networkCode="TX" stationCode="PB07" locationCode="00" channelCode="HN2"></waveformID>
        <phaseHint>S</phaseHint>
        <evaluationMode>manual</evaluationMode>
        <creationInfo>
          <agencyID>TXNet</agencyID>
          <author>daniel</author>
          <creationTime>2022-11-17T16:48:26.533333Z</creationTime>
        </creationInfo>
      </pick>
      <pick publicID="smi:org.gfz-potsdam.de/geofon/Pick/20221117164826.533355.115012">
        <time>
          <value>2022-11-16T21:32:50.722062Z</value>
        </time>
        <waveformID networkCode="TX" stationCode="PB07" locationCode="00" channelCode="HNZ"></waveformID>
        <phaseHint>P</phaseHint>
        <evaluationMode>manual</evaluationMode>
        <creationInfo>
          <agencyID>TXNet</agencyID>
          <author>daniel</author>
          <creationTime>2022-11-17T16:48:26.533369Z</creationTime>
        </creationInfo>
      </pick>
      <pick publicID="smi:org.gfz-potsdam.de/geofon/Pick/20221117165346.006932.115338">
        <time>
          <value>2022-11-16T21:32:47.826042Z</value>
          <lowerUncertainty>0.1000000015</lowerUncertainty>
          <upperUncertainty>0.1000000015</upperUncertainty>
        </time>
        <waveformID networkCode="TX" stationCode="PB40" locationCode="00" channelCode="HN2"></waveformID>
        <phaseHint>S</phaseHint>
        <evaluationMode>manual</evaluationMode>
        <creationInfo>
          <agencyID>TXNet</agencyID>
          <author>daniel</author>
          <creationTime>2022-11-17T16:53:46.006988Z</creationTime>
        </creationInfo>
      </pick>
      <pick publicID="smi:org.gfz-potsdam.de/geofon/Pick/20221117165346.007624.115339">
        <time>
          <value>2022-11-16T21:32:50.899369Z</value>
        </time>
        <waveformID networkCode="TX" stationCode="PB11" locationCode="00" channelCode="HNZ"></waveformID>
        <phaseHint>P</phaseHint>
        <evaluationMode>manual</evaluationMode>
        <creationInfo>
          <agencyID>TXNet</agencyID>
          <author>daniel</author>
          <creationTime>2022-11-17T16:53:46.007642Z</creationTime>
        </creationInfo>
      </pick>
      <pick publicID="smi:org.gfz-potsdam.de/geofon/Pick/20221117165346.00767.115340">
        <time>
          <value>2022-11-16T21:32:55.525478Z</value>
        </time>
        <waveformID networkCode="TX" stationCode="PB11" locationCode="00" channelCode="HN2"></waveformID>
        <phaseHint>S</phaseHint>
        <evaluationMode>manual</evaluationMode>
        <creationInfo>
          <agencyID>TXNet</agencyID>
          <author>daniel</author>
          <creationTime>2022-11-17T16:53:46.007686Z</creationTime>
        </creationInfo>
      </pick>
      <pick publicID="smi:org.gfz-potsdam.de/geofon/Pick/20221117165346.007912.115341">
        <time>
          <value>2022-11-16T21:32:52.005122Z</value>
          <lowerUncertainty>0.1000000015</lowerUncertainty>
          <upperUncertainty>0.1000000015</upperUncertainty>
        </time>
        <waveformID networkCode="TX" stationCode="PB36" locationCode="00" channelCode="HN1"></waveformID>
        <phaseHint>P</phaseHint>
        <evaluationMode>manual</evaluationMode>
        <creationInfo>
          <agencyID>TXNet</agencyID>
          <author>daniel</author>
          <creationTime>2022-11-17T16:53:46.007927Z</creationTime>
        </creationInfo>
      </pick>
      <pick publicID="smi:org.gfz-potsdam.de/geofon/Pick/20221117165346.00795.115342">
        <time>
          <value>2022-11-16T21:32:56.977910Z</value>
          <lowerUncertainty>0.3000000119</lowerUncertainty>
          <upperUncertainty>0.3000000119</upperUncertainty>
        </time>
        <waveformID networkCode="TX" stationCode="PB36" locationCode="00" channelCode="HN2"></waveformID>
        <phaseHint>S</phaseHint>
        <evaluationMode>manual</evaluationMode>
        <creationInfo>
          <agencyID>TXNet</agencyID>
          <author>daniel</author>
          <creationTime>2022-11-17T16:53:46.007965Z</creationTime>
        </creationInfo>
      </pick>
      <pick publicID="smi:org.gfz-potsdam.de/geofon/Pick/20221117165747.528372.115574">
        <time>
          <value>2022-11-16T21:33:01.789583Z</value>
          <lowerUncertainty>0.3000000119</lowerUncertainty>
          <upperUncertainty>0.3000000119</upperUncertainty>
        </time>
        <waveformID networkCode="TX" stationCode="DG06" locationCode="00" channelCode="CH2"></waveformID>
        <phaseHint>S</phaseHint>
        <evaluationMode>manual</evaluationMode>
        <creationInfo>
          <agencyID>TXNet</agencyID>
          <author>daniel</author>
          <creationTime>2022-11-17T16:57:47.528444Z</creationTime>
        </creationInfo>
      </pick>
      <pick publicID="smi:org.gfz-potsdam.de/geofon/Pick/20221117165747.528533.115575">
        <time>
          <value>2022-11-16T21:33:01.328883Z</value>
          <lowerUncertainty>0.3000000119</lowerUncertainty>
          <upperUncertainty>0.3000000119</upperUncertainty>
        </time>
        <waveformID networkCode="TX" stationCode="DG08" locationCode="00" channelCode="CH1"></waveformID>
        <phaseHint>S</phaseHint>
        <evaluationMode>manual</evaluationMode>
        <creationInfo>
          <agencyID>TXNet</agencyID>
          <author>daniel</author>
          <creationTime>2022-11-17T16:57:47.528557Z</creationTime>
        </creationInfo>
      </pick>
      <pick publicID="smi:org.gfz-potsdam.de/geofon/Pick/20221117165747.52863.115576">
        <time>
          <value>2022-11-16T21:33:02.019112Z</value>
          <lowerUncertainty>0.3000000119</lowerUncertainty>
          <upperUncertainty>0.3000000119</upperUncertainty>
        </time>
        <waveformID networkCode="TX" stationCode="DG03" locationCode="00" channelCode="CH2"></waveformID>
        <phaseHint>S</phaseHint>
        <evaluationMode>manual</evaluationMode>
        <creationInfo>
          <agencyID>TXNet</agencyID>
          <author>daniel</author>
          <creationTime>2022-11-17T16:57:47.528646Z</creationTime>
        </creationInfo>
      </pick>
      <pick publicID="smi:org.gfz-potsdam.de/geofon/Pick/20221117165747.528694.115577">
        <time>
          <value>2022-11-16T21:33:01.800769Z</value>
          <lowerUncertainty>0.3000000119</lowerUncertainty>
          <upperUncertainty>0.3000000119</upperUncertainty>
        </time>
        <waveformID networkCode="GM" stationCode="NMP41" locationCode="01" channelCode="HHE"></waveformID>
        <phaseHint>S</phaseHint>
        <evaluationMode>manual</evaluationMode>
        <creationInfo>
          <agencyID>TXNet</agencyID>
          <author>daniel</author>
          <creationTime>2022-11-17T16:57:47.528709Z</creationTime>
        </creationInfo>
      </pick>
      <pick publicID="smi:org.gfz-potsdam.de/geofon/Pick/20221117165747.528778.115578">
        <time>
          <value>2022-11-16T21:33:02.068442Z</value>
          <lowerUncertainty>0.3000000119</lowerUncertainty>
          <upperUncertainty>0.3000000119</upperUncertainty>
        </time>
        <waveformID networkCode="4O" stationCode="LWM2" locationCode="0K" channelCode="HHE"></waveformID>
        <phaseHint>S</phaseHint>
        <evaluationMode>manual</evaluationMode>
        <creationInfo>
          <agencyID>TXNet</agencyID>
          <author>daniel</author>
          <creationTime>2022-11-17T16:57:47.528794Z</creationTime>
        </creationInfo>
      </pick>
      <pick publicID="smi:org.gfz-potsdam.de/geofon/Pick/20221117165747.528842.115579">
        <time>
          <value>2022-11-16T21:33:02.195239Z</value>
          <lowerUncertainty>0.3000000119</lowerUncertainty>
          <upperUncertainty>0.3000000119</upperUncertainty>
        </time>
        <waveformID networkCode="4O" stationCode="LWM1" locationCode="0K" channelCode="HHE"></waveformID>
        <phaseHint>S</phaseHint>
        <evaluationMode>manual</evaluationMode>
        <creationInfo>
          <agencyID>TXNet</agencyID>
          <author>daniel</author>
          <creationTime>2022-11-17T16:57:47.528856Z</creationTime>
        </creationInfo>
      </pick>
      <pick publicID="smi:org.gfz-potsdam.de/geofon/Pick/20221117165747.528881.115580">
        <time>
          <value>2022-11-16T21:33:03.768633Z</value>
          <lowerUncertainty>0.3000000119</lowerUncertainty>
          <upperUncertainty>0.3000000119</upperUncertainty>
        </time>
        <waveformID networkCode="SC" stationCode="PDB" locationCode="00" channelCode="HHE"></waveformID>
        <phaseHint>S</phaseHint>
        <evaluationMode>manual</evaluationMode>
        <creationInfo>
          <agencyID>TXNet</agencyID>
          <author>daniel</author>
          <creationTime>2022-11-17T16:57:47.528896Z</creationTime>
        </creationInfo>
      </pick>
      <pick publicID="smi:org.gfz-potsdam.de/geofon/Pick/20221117165747.52892.115581">
        <time>
          <value>2022-11-16T21:32:55.459277Z</value>
          <lowerUncertainty>0.1000000015</lowerUncertainty>
          <upperUncertainty>0.1000000015</upperUncertainty>
        </time>
        <waveformID networkCode="SC" stationCode="PDB" locationCode="00" channelCode="HHN"></waveformID>
        <phaseHint>P</phaseHint>
        <evaluationMode>manual</evaluationMode>
        <creationInfo>
          <agencyID>TXNet</agencyID>
          <author>daniel</author>
          <creationTime>2022-11-17T16:57:47.528934Z</creationTime>
        </creationInfo>
      </pick>
      <pick publicID="smi:org.gfz-potsdam.de/geofon/Pick/20221117165747.528958.115582">
        <time>
          <value>2022-11-16T21:33:04.189830Z</value>
          <lowerUncertainty>0.3000000119</lowerUncertainty>
          <upperUncertainty>0.3000000119</upperUncertainty>
        </time>
        <waveformID networkCode="GM" stationCode="NMP23" locationCode="01" channelCode="HHE"></waveformID>
        <phaseHint>S</phaseHint>
        <evaluationMode>manual</evaluationMode>
        <creationInfo>
          <agencyID>TXNet</agencyID>
          <author>daniel</author>
          <creationTime>2022-11-17T16:57:47.528973Z</creationTime>
        </creationInfo>
      </pick>
      <pick publicID="smi:org.gfz-potsdam.de/geofon/Pick/20221117165747.528997.115583">
        <time>
          <value>2022-11-16T21:32:55.528667Z</value>
        </time>
        <waveformID networkCode="GM" stationCode="NMP23" locationCode="01" channelCode="HHZ"></waveformID>
        <phaseHint>P</phaseHint>
        <evaluationMode>manual</evaluationMode>
        <creationInfo>
          <agencyID>TXNet</agencyID>
          <author>daniel</author>
          <creationTime>2022-11-17T16:57:47.529012Z</creationTime>
        </creationInfo>
      </pick>
      <pick publicID="smi:org.gfz-potsdam.de/geofon/Pick/20221117165747.529072.115584">
        <time>
          <value>2022-11-16T21:33:04.834678Z</value>
        </time>
        <waveformID networkCode="TX" stationCode="DG04" locationCode="00" channelCode="CH1"></waveformID>
        <phaseHint>S</phaseHint>
        <evaluationMode>manual</evaluationMode>
        <creationInfo>
          <agencyID>TXNet</agencyID>
          <author>daniel</author>
          <creationTime>2022-11-17T16:57:47.529088Z</creationTime>
        </creationInfo>
      </pick>
      <pick publicID="smi:org.gfz-potsdam.de/geofon/Pick/20221117165747.529145.115585">
        <time>
          <value>2022-11-16T21:33:05.178606Z</value>
        </time>
        <waveformID networkCode="GM" stationCode="NMP51" locationCode="01" channelCode="HHN"></waveformID>
        <phaseHint>S</phaseHint>
        <evaluationMode>manual</evaluationMode>
        <creationInfo>
          <agencyID>TXNet</agencyID>
          <author>daniel</author>
          <creationTime>2022-11-17T16:57:47.529161Z</creationTime>
        </creationInfo>
      </pick>
      <pick publicID="smi:org.gfz-potsdam.de/geofon/Pick/20221117165747.529448.115586">
        <time>
          <value>2022-11-16T21:33:01.729035Z</value>
        </time>
        <waveformID networkCode="TX" stationCode="PB05" locationCode="00" channelCode="HHZ"></waveformID>
        <phaseHint>P</phaseHint>
        <evaluationMode>manual</evaluationMode>
        <creationInfo>
          <agencyID>TXNet</agencyID>
          <author>daniel</author>
          <creationTime>2022-11-17T16:57:47.529464Z</creationTime>
        </creationInfo>
      </pick>
      <pick publicID="smi:org.gfz-potsdam.de/geofon/Pick/20221117165747.529487.115587">
        <time>
          <value>2022-11-16T21:33:14.327090Z</value>
        </time>
        <waveformID networkCode="TX" stationCode="PB05" locationCode="00" channelCode="HH1"></waveformID>
        <phaseHint>S</phaseHint>
        <evaluationMode>manual</evaluationMode>
        <creationInfo>
          <agencyID>TXNet</agencyID>
          <author>daniel</author>
          <creationTime>2022-11-17T16:57:47.529501Z</creationTime>
        </creationInfo>
      </pick>
      <pick publicID="smi:org.gfz-potsdam.de/geofon/Pick/20221117165747.529527.115588">
        <time>
          <value>2022-11-16T21:33:02.281820Z</value>
          <lowerUncertainty>0.1000000015</lowerUncertainty>
          <upperUncertainty>0.1000000015</upperUncertainty>
        </time>
        <waveformID networkCode="4O" stationCode="DB03" channelCode="HHZ"></waveformID>
        <phaseHint>P</phaseHint>
        <polarity>positive</polarity>
        <evaluationMode>manual</evaluationMode>
        <creationInfo>
          <agencyID>TXNet</agencyID>
          <author>daniel</author>
          <creationTime>2022-11-17T16:57:47.529542Z</creationTime>
        </creationInfo>
      </pick>
      <pick publicID="smi:org.gfz-potsdam.de/geofon/Pick/20221117165747.52956.115589">
        <time>
          <value>2022-11-16T21:33:15.457845Z</value>
        </time>
        <waveformID networkCode="4O" stationCode="DB03" channelCode="HHE"></waveformID>
        <phaseHint>S</phaseHint>
        <evaluationMode>manual</evaluationMode>
        <creationInfo>
          <agencyID>TXNet</agencyID>
          <author>daniel</author>
          <creationTime>2022-11-17T16:57:47.529576Z</creationTime>
        </creationInfo>
      </pick>
      <pick publicID="smi:org.gfz-potsdam.de/geofon/Pick/20221117165747.529606.115590">
        <time>
          <value>2022-11-16T21:33:01.373076Z</value>
          <lowerUncertainty>0.1000000015</lowerUncertainty>
          <upperUncertainty>0.1000000015</upperUncertainty>
        </time>
        <waveformID networkCode="TX" stationCode="PB22" locationCode="00" channelCode="HHZ"></waveformID>
        <phaseHint>P</phaseHint>
        <polarity>positive</polarity>
        <evaluationMode>manual</evaluationMode>
        <creationInfo>
          <agencyID>TXNet</agencyID>
          <author>daniel</author>
          <creationTime>2022-11-17T16:57:47.529620Z</creationTime>
        </creationInfo>
      </pick>
      <pick publicID="smi:org.gfz-potsdam.de/geofon/Pick/20221117165747.529639.115591">
        <time>
          <value>2022-11-16T21:33:15.295201Z</value>
          <lowerUncertainty>0.3000000119</lowerUncertainty>
          <upperUncertainty>0.3000000119</upperUncertainty>
        </time>
        <waveformID networkCode="TX" stationCode="PB22" locationCode="00" channelCode="HH1"></waveformID>
        <phaseHint>S</phaseHint>
        <evaluationMode>manual</evaluationMode>
        <creationInfo>
          <agencyID>TXNet</agencyID>
          <author>daniel</author>
          <creationTime>2022-11-17T16:57:47.529655Z</creationTime>
        </creationInfo>
      </pick>
      <pick publicID="smi:org.gfz-potsdam.de/geofon/Pick/20221117165747.52968.115592">
        <time>
          <value>2022-11-16T21:33:02.442838Z</value>
        </time>
        <waveformID networkCode="4O" stationCode="DB04" channelCode="HHZ"></waveformID>
        <phaseHint>P</phaseHint>
        <evaluationMode>manual</evaluationMode>
        <creationInfo>
          <agencyID>TXNet</agencyID>
          <author>daniel</author>
          <creationTime>2022-11-17T16:57:47.529694Z</creationTime>
        </creationInfo>
      </pick>
    </event>
  </eventParameters>
</q:quakeml>
