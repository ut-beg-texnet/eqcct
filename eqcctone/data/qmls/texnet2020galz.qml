<?xml version='1.0' encoding='utf-8'?>
<q:quakeml xmlns="http://quakeml.org/xmlns/bed/1.2" xmlns:q="http://quakeml.org/xmlns/quakeml/1.2">
  <eventParameters publicID="smi:local/3e56f1bc-b41a-4ac8-a8ec-c19f0ef118c3">
    <event publicID="smi:org.gfz-potsdam.de/geofon/texnet2020galz">
      <preferredOriginID>smi:org.gfz-potsdam.de/geofon/NLL.20200416191747.121847.63133</preferredOriginID>
      <preferredMagnitudeID>smi:org.gfz-potsdam.de/geofon/NLL.20200416191747.121847.63133/netMag/ML(TexNet)</preferredMagnitudeID>
      <preferredFocalMechanismID>smi:org.gfz-potsdam.de/geofon/FocalMechanism/20200327224033.238024.163278</preferredFocalMechanismID>
      <type>earthquake</type>
      <typeCertainty>known</typeCertainty>
      <description>
        <text>Western Texas</text>
        <type>region name</type>
      </description>
      <creationInfo>
        <agencyID>TXNet</agencyID>
        <author>scevent@sc3primary.beg.utexas.edu</author>
        <creationTime>2020-03-26T15:16:42.659244Z</creationTime>
      </creationInfo>
      <origin publicID="smi:org.gfz-potsdam.de/geofon/NLL.20200416191747.121847.63133">
        <time>
          <value>2020-03-26T15:16:27.844763Z</value>
        </time>
        <latitude>
          <value>31.70250443</value>
          <uncertainty>0.4829126</uncertainty>
        </latitude>
        <longitude>
          <value>-104.0521372</value>
          <uncertainty>0.5536388507</uncertainty>
        </longitude>
        <depth>
          <value>6157.275391</value>
          <uncertainty>1900.152834</uncertainty>
        </depth>
        <methodID>smi:org.gfz-potsdam.de/geofon/NonLinLoc</methodID>
        <earthModelID>smi:org.gfz-potsdam.de/geofon/PB1D-20170918-topoLV</earthModelID>
        <quality>
          <associatedPhaseCount>84</associatedPhaseCount>
          <usedPhaseCount>48</usedPhaseCount>
          <usedStationCount>31</usedStationCount>
          <standardError>0.2815587328</standardError>
          <azimuthalGap>49.353744</azimuthalGap>
          <secondaryAzimuthalGap>93.10250305</secondaryAzimuthalGap>
          <groundTruthLevel>-</groundTruthLevel>
          <minimumDistance>0.224160181</minimumDistance>
          <maximumDistance>0.795902985</maximumDistance>
          <medianDistance>0.5814609527</medianDistance>
        </quality>
        <evaluationMode>manual</evaluationMode>
        <evaluationStatus>final</evaluationStatus>
        <creationInfo>
          <agencyID>TXNet</agencyID>
          <author>Tricia</author>
          <creationTime>2020-04-16T19:17:48.920720Z</creationTime>
        </creationInfo>
        <originUncertainty>
          <horizontalUncertainty>913.7805628</horizontalUncertainty>
          <minHorizontalUncertainty>637.4655775</minHorizontalUncertainty>
          <maxHorizontalUncertainty>913.7805628</maxHorizontalUncertainty>
          <azimuthMaxHorizontalUncertainty>56.58254237</azimuthMaxHorizontalUncertainty>
          <confidenceEllipsoid>
            <semiMajorAxisLength>3577269.811</semiMajorAxisLength>
            <semiMinorAxisLength>786440.268</semiMinorAxisLength>
            <semiIntermediateAxisLength>1111402.794</semiIntermediateAxisLength>
            <majorAxisPlunge>3.816643974</majorAxisPlunge>
            <majorAxisAzimuth>-163.9444627</majorAxisAzimuth>
            <majorAxisRotation>91.22645971</majorAxisRotation>
          </confidenceEllipsoid>
        </originUncertainty>
        <arrival publicID="smi:org.gfz-potsdam.de/geofon/Pick/20200326152204.026602.256215_NLL.20200416191747.121847.63133">
          <pickID>smi:org.gfz-potsdam.de/geofon/Pick/20200326152204.026602.256215</pickID>
          <phase>P</phase>
          <timeCorrection>0.0</timeCorrection>
          <azimuth>110.7805778</azimuth>
          <distance>0.3500816135</distance>
          <takeoffAngle>
            <value>82.0</value>
          </takeoffAngle>
          <timeResidual>0.195709865</timeResidual>
          <timeWeight>1.176902854</timeWeight>
        </arrival>
        <arrival publicID="smi:org.gfz-potsdam.de/geofon/Pick/20200326152204.026802.256217_NLL.20200416191747.121847.63133">
          <pickID>smi:org.gfz-potsdam.de/geofon/Pick/20200326152204.026802.256217</pickID>
          <phase>P</phase>
          <timeCorrection>0.0</timeCorrection>
          <azimuth>154.7843702</azimuth>
          <distance>0.3682233867</distance>
          <takeoffAngle>
            <value>81.9</value>
          </takeoffAngle>
          <timeResidual>-0.1210145258</timeResidual>
          <timeWeight>1.225421863</timeWeight>
        </arrival>
        <arrival publicID="smi:org.gfz-potsdam.de/geofon/Pick/20200326152204.026844.256218_NLL.20200416191747.121847.63133">
          <pickID>smi:org.gfz-potsdam.de/geofon/Pick/20200326152204.026844.256218</pickID>
          <phase>P</phase>
          <timeCorrection>0.0</timeCorrection>
          <azimuth>148.9769613</azimuth>
          <distance>0.4905716128</distance>
          <takeoffAngle>
            <value>81.7</value>
          </takeoffAngle>
          <timeResidual>-0.08404994439</timeResidual>
          <timeWeight>1.224036825</timeWeight>
        </arrival>
        <arrival publicID="smi:org.gfz-potsdam.de/geofon/Pick/20200326152204.026887.256219_NLL.20200416191747.121847.63133">
          <pickID>smi:org.gfz-potsdam.de/geofon/Pick/20200326152204.026887.256219</pickID>
          <phase>P</phase>
          <timeCorrection>0.0</timeCorrection>
          <azimuth>122.5603041</azimuth>
          <distance>0.5481873156</distance>
          <takeoffAngle>
            <value>81.7</value>
          </takeoffAngle>
          <timeResidual>0.07918253759</timeResidual>
          <timeWeight>1.08206638</timeWeight>
        </arrival>
        <arrival publicID="smi:org.gfz-potsdam.de/geofon/Pick/20200326152204.026919.256220_NLL.20200416191747.121847.63133">
          <pickID>smi:org.gfz-potsdam.de/geofon/Pick/20200326152204.026919.256220</pickID>
          <phase>S</phase>
          <timeCorrection>0.0</timeCorrection>
          <azimuth>122.5603041</azimuth>
          <distance>0.5481873156</distance>
          <takeoffAngle>
            <value>81.7</value>
          </takeoffAngle>
          <timeResidual>-0.1209269936</timeResidual>
          <timeWeight>1.083926293</timeWeight>
        </arrival>
        <arrival publicID="smi:org.gfz-potsdam.de/geofon/Pick/20200326152204.026952.256221_NLL.20200416191747.121847.63133">
          <pickID>smi:org.gfz-potsdam.de/geofon/Pick/20200326152204.026952.256221</pickID>
          <phase>S</phase>
          <timeCorrection>0.0</timeCorrection>
          <azimuth>122.99249</azimuth>
          <distance>0.5761798089</distance>
          <takeoffAngle>
            <value>81.6</value>
          </takeoffAngle>
          <timeResidual>-0.2880656391</timeResidual>
          <timeWeight>1.032329525</timeWeight>
        </arrival>
        <arrival publicID="smi:org.gfz-potsdam.de/geofon/Pick/20200326152204.026985.256222_NLL.20200416191747.121847.63133">
          <pickID>smi:org.gfz-potsdam.de/geofon/Pick/20200326152204.026985.256222</pickID>
          <phase>P</phase>
          <timeCorrection>0.0</timeCorrection>
          <azimuth>122.99249</azimuth>
          <distance>0.5761798089</distance>
          <takeoffAngle>
            <value>81.7</value>
          </takeoffAngle>
          <timeResidual>0.1859723624</timeResidual>
          <timeWeight>1.102246009</timeWeight>
        </arrival>
        <arrival publicID="smi:org.gfz-potsdam.de/geofon/Pick/20200326152204.027024.256223_NLL.20200416191747.121847.63133">
          <pickID>smi:org.gfz-potsdam.de/geofon/Pick/20200326152204.027024.256223</pickID>
          <phase>P</phase>
          <timeCorrection>0.0</timeCorrection>
          <azimuth>170.8810044</azimuth>
          <distance>0.4966281256</distance>
          <takeoffAngle>
            <value>81.7</value>
          </takeoffAngle>
          <timeResidual>-0.1338891651</timeResidual>
          <timeWeight>1.215316903</timeWeight>
        </arrival>
        <arrival publicID="smi:org.gfz-potsdam.de/geofon/Pick/20200326152204.027057.256224_NLL.20200416191747.121847.63133">
          <pickID>smi:org.gfz-potsdam.de/geofon/Pick/20200326152204.027057.256224</pickID>
          <phase>S</phase>
          <timeCorrection>0.0</timeCorrection>
          <azimuth>170.8810044</azimuth>
          <distance>0.4966281256</distance>
          <takeoffAngle>
            <value>81.7</value>
          </takeoffAngle>
          <timeResidual>-0.4160116802</timeResidual>
          <timeWeight>0.9640854236</timeWeight>
        </arrival>
        <arrival publicID="smi:org.gfz-potsdam.de/geofon/Pick/20200326152204.027216.256226_NLL.20200416191747.121847.63133">
          <pickID>smi:org.gfz-potsdam.de/geofon/Pick/20200326152204.027216.256226</pickID>
          <phase>S</phase>
          <timeCorrection>0.0</timeCorrection>
          <azimuth>4.012871371</azimuth>
          <distance>0.233892358</distance>
          <takeoffAngle>
            <value>82.5</value>
          </takeoffAngle>
          <timeResidual>-0.2595788853</timeResidual>
          <timeWeight>0.9919880986</timeWeight>
        </arrival>
        <arrival publicID="smi:org.gfz-potsdam.de/geofon/Pick/20200326152204.027277.256228_NLL.20200416191747.121847.63133">
          <pickID>smi:org.gfz-potsdam.de/geofon/Pick/20200326152204.027277.256228</pickID>
          <phase>P</phase>
          <timeCorrection>0.0</timeCorrection>
          <azimuth>116.0686531</azimuth>
          <distance>0.7417623981</distance>
          <takeoffAngle>
            <value>79.0</value>
          </takeoffAngle>
          <timeResidual>-0.0182809371</timeResidual>
          <timeWeight>1.099672303</timeWeight>
        </arrival>
        <arrival publicID="smi:org.gfz-potsdam.de/geofon/Pick/20200326152204.02731.256229_NLL.20200416191747.121847.63133">
          <pickID>smi:org.gfz-potsdam.de/geofon/Pick/20200326152204.02731.256229</pickID>
          <phase>P</phase>
          <timeCorrection>0.0</timeCorrection>
          <azimuth>124.2280845</azimuth>
          <distance>0.7529646196</distance>
          <takeoffAngle>
            <value>77.1</value>
          </takeoffAngle>
          <timeResidual>-0.02458800061</timeResidual>
          <timeWeight>1.002333553</timeWeight>
        </arrival>
        <arrival publicID="smi:org.gfz-potsdam.de/geofon/Pick/20200326152204.027345.256230_NLL.20200416191747.121847.63133">
          <pickID>smi:org.gfz-potsdam.de/geofon/Pick/20200326152204.027345.256230</pickID>
          <phase>S</phase>
          <timeCorrection>0.0</timeCorrection>
          <azimuth>124.2280845</azimuth>
          <distance>0.7529646196</distance>
          <takeoffAngle>
            <value>77.4</value>
          </takeoffAngle>
          <timeResidual>-0.4087711979</timeResidual>
          <timeWeight>0.0</timeWeight>
        </arrival>
        <arrival publicID="smi:org.gfz-potsdam.de/geofon/Pick/20200326152313.186867.256298_NLL.20200416191747.121847.63133">
          <pickID>smi:org.gfz-potsdam.de/geofon/Pick/20200326152313.186867.256298</pickID>
          <phase>P</phase>
          <timeCorrection>0.0</timeCorrection>
          <azimuth>265.2984552</azimuth>
          <distance>0.3839104888</distance>
          <takeoffAngle>
            <value>81.9</value>
          </takeoffAngle>
          <timeResidual>0.3002648438</timeResidual>
          <timeWeight>0.8733140939</timeWeight>
        </arrival>
        <arrival publicID="smi:org.gfz-potsdam.de/geofon/Pick/20200326152313.187024.256299_NLL.20200416191747.121847.63133">
          <pickID>smi:org.gfz-potsdam.de/geofon/Pick/20200326152313.187024.256299</pickID>
          <phase>S</phase>
          <timeCorrection>0.0</timeCorrection>
          <azimuth>265.2984552</azimuth>
          <distance>0.3839104888</distance>
          <takeoffAngle>
            <value>81.9</value>
          </takeoffAngle>
          <timeResidual>-0.4758142045</timeResidual>
          <timeWeight>0.9251397593</timeWeight>
        </arrival>
        <arrival publicID="smi:org.gfz-potsdam.de/geofon/Pick/20200326152313.187063.256300_NLL.20200416191747.121847.63133">
          <pickID>smi:org.gfz-potsdam.de/geofon/Pick/20200326152313.187063.256300</pickID>
          <phase>S</phase>
          <timeCorrection>0.0</timeCorrection>
          <azimuth>277.6950513</azimuth>
          <distance>0.3971195901</distance>
          <takeoffAngle>
            <value>81.8</value>
          </takeoffAngle>
          <timeResidual>-0.515430975</timeResidual>
          <timeWeight>0.8911264778</timeWeight>
        </arrival>
        <arrival publicID="smi:org.gfz-potsdam.de/geofon/Pick/20200326152313.187094.256301_NLL.20200416191747.121847.63133">
          <pickID>smi:org.gfz-potsdam.de/geofon/Pick/20200326152313.187094.256301</pickID>
          <phase>P</phase>
          <timeCorrection>0.0</timeCorrection>
          <azimuth>277.6950513</azimuth>
          <distance>0.3971195901</distance>
          <takeoffAngle>
            <value>81.8</value>
          </takeoffAngle>
          <timeResidual>0.3041654295</timeResidual>
          <timeWeight>0.9288986402</timeWeight>
        </arrival>
        <arrival publicID="smi:org.gfz-potsdam.de/geofon/Pick/20200326152313.187125.256302_NLL.20200416191747.121847.63133">
          <pickID>smi:org.gfz-potsdam.de/geofon/Pick/20200326152313.187125.256302</pickID>
          <phase>P</phase>
          <timeCorrection>0.0</timeCorrection>
          <azimuth>22.92160887</azimuth>
          <distance>0.4201319636</distance>
          <takeoffAngle>
            <value>81.8</value>
          </takeoffAngle>
          <timeResidual>0.1020932844</timeResidual>
          <timeWeight>1.175705793</timeWeight>
        </arrival>
        <arrival publicID="smi:org.gfz-potsdam.de/geofon/Pick/20200326152313.187152.256303_NLL.20200416191747.121847.63133">
          <pickID>smi:org.gfz-potsdam.de/geofon/Pick/20200326152313.187152.256303</pickID>
          <phase>S</phase>
          <timeCorrection>0.0</timeCorrection>
          <azimuth>22.92160887</azimuth>
          <distance>0.4201319636</distance>
          <takeoffAngle>
            <value>81.8</value>
          </takeoffAngle>
          <timeResidual>0.08068189561</timeResidual>
          <timeWeight>1.068226079</timeWeight>
        </arrival>
        <arrival publicID="smi:org.gfz-potsdam.de/geofon/Pick/20200326152313.187183.256304_NLL.20200416191747.121847.63133">
          <pickID>smi:org.gfz-potsdam.de/geofon/Pick/20200326152313.187183.256304</pickID>
          <phase>S</phase>
          <timeCorrection>0.0</timeCorrection>
          <azimuth>148.9769613</azimuth>
          <distance>0.4905716128</distance>
          <takeoffAngle>
            <value>81.7</value>
          </takeoffAngle>
          <timeResidual>-0.04330808606</timeResidual>
          <timeWeight>1.09188406</timeWeight>
        </arrival>
        <arrival publicID="smi:org.gfz-potsdam.de/geofon/Pick/20200326152313.187217.256305_NLL.20200416191747.121847.63133">
          <pickID>smi:org.gfz-potsdam.de/geofon/Pick/20200326152313.187217.256305</pickID>
          <phase>S</phase>
          <timeCorrection>0.0</timeCorrection>
          <azimuth>18.14770822</azimuth>
          <distance>0.5288563627</distance>
          <takeoffAngle>
            <value>81.7</value>
          </takeoffAngle>
          <timeResidual>-0.3349962726</timeResidual>
          <timeWeight>1.013086976</timeWeight>
        </arrival>
        <arrival publicID="smi:org.gfz-potsdam.de/geofon/Pick/20200326152313.187244.256306_NLL.20200416191747.121847.63133">
          <pickID>smi:org.gfz-potsdam.de/geofon/Pick/20200326152313.187244.256306</pickID>
          <phase>P</phase>
          <timeCorrection>0.0</timeCorrection>
          <azimuth>18.14770822</azimuth>
          <distance>0.5288563627</distance>
          <takeoffAngle>
            <value>81.7</value>
          </takeoffAngle>
          <timeResidual>-0.06445016235</timeResidual>
          <timeWeight>1.117464294</timeWeight>
        </arrival>
        <arrival publicID="smi:org.gfz-potsdam.de/geofon/Pick/20200326152313.187274.256307_NLL.20200416191747.121847.63133">
          <pickID>smi:org.gfz-potsdam.de/geofon/Pick/20200326152313.187274.256307</pickID>
          <phase>P</phase>
          <timeCorrection>0.0</timeCorrection>
          <azimuth>46.43214937</azimuth>
          <distance>0.5359472567</distance>
          <takeoffAngle>
            <value>81.7</value>
          </takeoffAngle>
          <timeResidual>0.254650933</timeResidual>
          <timeWeight>0.9690130284</timeWeight>
        </arrival>
        <arrival publicID="smi:org.gfz-potsdam.de/geofon/Pick/20200326152313.187308.256308_NLL.20200416191747.121847.63133">
          <pickID>smi:org.gfz-potsdam.de/geofon/Pick/20200326152313.187308.256308</pickID>
          <phase>P</phase>
          <timeCorrection>0.0</timeCorrection>
          <azimuth>332.3932376</azimuth>
          <distance>0.5650254947</distance>
          <takeoffAngle>
            <value>81.7</value>
          </takeoffAngle>
          <timeResidual>-0.2966547582</timeResidual>
          <timeWeight>1.054811646</timeWeight>
        </arrival>
        <arrival publicID="smi:org.gfz-potsdam.de/geofon/Pick/20200326152358.946492.256577_NLL.20200416191747.121847.63133">
          <pickID>smi:org.gfz-potsdam.de/geofon/Pick/20200326152358.946492.256577</pickID>
          <phase>P</phase>
          <timeCorrection>0.0</timeCorrection>
          <azimuth>132.5731125</azimuth>
          <distance>1.045454945</distance>
          <takeoffAngle>
            <value>72.6</value>
          </takeoffAngle>
          <timeResidual>-0.4048810152</timeResidual>
          <timeWeight>0.0</timeWeight>
        </arrival>
        <arrival publicID="smi:org.gfz-potsdam.de/geofon/Pick/20200326152358.946643.256578_NLL.20200416191747.121847.63133">
          <pickID>smi:org.gfz-potsdam.de/geofon/Pick/20200326152358.946643.256578</pickID>
          <phase>P</phase>
          <timeCorrection>0.0</timeCorrection>
          <azimuth>270.4008298</azimuth>
          <distance>1.133449543</distance>
          <takeoffAngle>
            <value>72.6</value>
          </takeoffAngle>
          <timeResidual>-0.2289881786</timeResidual>
          <timeWeight>0.9678219992</timeWeight>
        </arrival>
        <arrival publicID="smi:org.gfz-potsdam.de/geofon/Pick/20200326152358.946685.256579_NLL.20200416191747.121847.63133">
          <pickID>smi:org.gfz-potsdam.de/geofon/Pick/20200326152358.946685.256579</pickID>
          <phase>P</phase>
          <timeCorrection>0.0</timeCorrection>
          <azimuth>108.1462602</azimuth>
          <distance>1.160328911</distance>
          <takeoffAngle>
            <value>72.6</value>
          </takeoffAngle>
          <timeResidual>-1.099300056</timeResidual>
          <timeWeight>0.0</timeWeight>
        </arrival>
        <arrival publicID="smi:org.gfz-potsdam.de/geofon/Pick/20200326152358.946718.256580_NLL.20200416191747.121847.63133">
          <pickID>smi:org.gfz-potsdam.de/geofon/Pick/20200326152358.946718.256580</pickID>
          <phase>P</phase>
          <timeCorrection>0.0</timeCorrection>
          <azimuth>221.5496962</azimuth>
          <distance>1.216620085</distance>
          <takeoffAngle>
            <value>72.6</value>
          </takeoffAngle>
          <timeResidual>-0.7912819388</timeResidual>
          <timeWeight>0.6320952365</timeWeight>
        </arrival>
        <arrival publicID="smi:org.gfz-potsdam.de/geofon/Pick/20200326153452.37426.258036_NLL.20200416191747.121847.63133">
          <pickID>smi:org.gfz-potsdam.de/geofon/Pick/20200326153452.37426.258036</pickID>
          <phase>P</phase>
          <timeCorrection>0.0</timeCorrection>
          <azimuth>4.012871371</azimuth>
          <distance>0.233892358</distance>
          <takeoffAngle>
            <value>82.5</value>
          </takeoffAngle>
          <timeResidual>0.1754803272</timeResidual>
          <timeWeight>1.205246435</timeWeight>
        </arrival>
        <arrival publicID="smi:org.gfz-potsdam.de/geofon/Pick/20200326153452.374363.258037_NLL.20200416191747.121847.63133">
          <pickID>smi:org.gfz-potsdam.de/geofon/Pick/20200326153452.374363.258037</pickID>
          <phase>P</phase>
          <timeCorrection>0.0</timeCorrection>
          <azimuth>130.3717009</azimuth>
          <distance>0.2303010636</distance>
          <takeoffAngle>
            <value>82.6</value>
          </takeoffAngle>
          <timeResidual>0.2616378922</timeResidual>
          <timeWeight>1.037313335</timeWeight>
        </arrival>
        <arrival publicID="smi:org.gfz-potsdam.de/geofon/Pick/20200326153452.374402.258038_NLL.20200416191747.121847.63133">
          <pickID>smi:org.gfz-potsdam.de/geofon/Pick/20200326153452.374402.258038</pickID>
          <phase>S</phase>
          <timeCorrection>0.0</timeCorrection>
          <azimuth>46.43214937</azimuth>
          <distance>0.5359472567</distance>
          <takeoffAngle>
            <value>81.7</value>
          </takeoffAngle>
          <timeResidual>-0.3227333103</timeResidual>
          <timeWeight>1.084720773</timeWeight>
        </arrival>
        <arrival publicID="smi:org.gfz-potsdam.de/geofon/Pick/20200326153452.374485.258039_NLL.20200416191747.121847.63133">
          <pickID>smi:org.gfz-potsdam.de/geofon/Pick/20200326153452.374485.258039</pickID>
          <phase>S</phase>
          <timeCorrection>0.0</timeCorrection>
          <azimuth>14.86802419</azimuth>
          <distance>0.5814609527</distance>
          <takeoffAngle>
            <value>81.6</value>
          </takeoffAngle>
          <timeResidual>-0.09538629784</timeResidual>
          <timeWeight>1.253614657</timeWeight>
        </arrival>
        <arrival publicID="smi:org.gfz-potsdam.de/geofon/Pick/20200326153452.37452.258040_NLL.20200416191747.121847.63133">
          <pickID>smi:org.gfz-potsdam.de/geofon/Pick/20200326153452.37452.258040</pickID>
          <phase>P</phase>
          <timeCorrection>0.0</timeCorrection>
          <azimuth>14.86802419</azimuth>
          <distance>0.5814609527</distance>
          <takeoffAngle>
            <value>81.7</value>
          </takeoffAngle>
          <timeResidual>-0.2047630565</timeResidual>
          <timeWeight>1.255183616</timeWeight>
        </arrival>
        <arrival publicID="smi:org.gfz-potsdam.de/geofon/Pick/20200326153452.374561.258041_NLL.20200416191747.121847.63133">
          <pickID>smi:org.gfz-potsdam.de/geofon/Pick/20200326153452.374561.258041</pickID>
          <phase>P</phase>
          <timeCorrection>0.0</timeCorrection>
          <azimuth>94.49521659</azimuth>
          <distance>0.7130223511</distance>
          <takeoffAngle>
            <value>81.3</value>
          </takeoffAngle>
          <timeResidual>-0.1634536367</timeResidual>
          <timeWeight>1.003913985</timeWeight>
        </arrival>
        <arrival publicID="smi:org.gfz-potsdam.de/geofon/Pick/20200326153452.374599.258042_NLL.20200416191747.121847.63133">
          <pickID>smi:org.gfz-potsdam.de/geofon/Pick/20200326153452.374599.258042</pickID>
          <phase>S</phase>
          <timeCorrection>0.0</timeCorrection>
          <azimuth>94.49521659</azimuth>
          <distance>0.7130223511</distance>
          <takeoffAngle>
            <value>81.3</value>
          </takeoffAngle>
          <timeResidual>-0.4132497761</timeResidual>
          <timeWeight>0.9937362193</timeWeight>
        </arrival>
        <arrival publicID="smi:org.gfz-potsdam.de/geofon/Pick/20200326153452.374629.258043_NLL.20200416191747.121847.63133">
          <pickID>smi:org.gfz-potsdam.de/geofon/Pick/20200326153452.374629.258043</pickID>
          <phase>P</phase>
          <timeCorrection>0.0</timeCorrection>
          <azimuth>11.76694989</azimuth>
          <distance>0.7338777561</distance>
          <takeoffAngle>
            <value>80.2</value>
          </takeoffAngle>
          <timeResidual>-0.6034287682</timeResidual>
          <timeWeight>0.7883622401</timeWeight>
        </arrival>
        <arrival publicID="smi:org.gfz-potsdam.de/geofon/Pick/20200326153452.374666.258044_NLL.20200416191747.121847.63133">
          <pickID>smi:org.gfz-potsdam.de/geofon/Pick/20200326153452.374666.258044</pickID>
          <phase>P</phase>
          <timeCorrection>0.0</timeCorrection>
          <azimuth>143.5215705</azimuth>
          <distance>0.7719056697</distance>
          <takeoffAngle>
            <value>74.4</value>
          </takeoffAngle>
          <timeResidual>-0.214006342</timeResidual>
          <timeWeight>0.0</timeWeight>
        </arrival>
        <arrival publicID="smi:org.gfz-potsdam.de/geofon/Pick/20200326153452.374697.258045_NLL.20200416191747.121847.63133">
          <pickID>smi:org.gfz-potsdam.de/geofon/Pick/20200326153452.374697.258045</pickID>
          <phase>S</phase>
          <timeCorrection>0.0</timeCorrection>
          <azimuth>143.5215705</azimuth>
          <distance>0.7719056697</distance>
          <takeoffAngle>
            <value>74.5</value>
          </takeoffAngle>
          <timeResidual>-0.6487666447</timeResidual>
          <timeWeight>0.0</timeWeight>
        </arrival>
        <arrival publicID="smi:org.gfz-potsdam.de/geofon/Pick/20200326153452.374731.258046_NLL.20200416191747.121847.63133">
          <pickID>smi:org.gfz-potsdam.de/geofon/Pick/20200326153452.374731.258046</pickID>
          <phase>P</phase>
          <timeCorrection>0.0</timeCorrection>
          <azimuth>163.2183909</azimuth>
          <distance>0.7946713786</distance>
          <takeoffAngle>
            <value>72.9</value>
          </takeoffAngle>
          <timeResidual>-0.7400411262</timeResidual>
          <timeWeight>0.0</timeWeight>
        </arrival>
        <arrival publicID="smi:org.gfz-potsdam.de/geofon/Pick/20200326153452.37476.258047_NLL.20200416191747.121847.63133">
          <pickID>smi:org.gfz-potsdam.de/geofon/Pick/20200326153452.37476.258047</pickID>
          <phase>S</phase>
          <timeCorrection>0.0</timeCorrection>
          <azimuth>163.2183909</azimuth>
          <distance>0.7946713786</distance>
          <takeoffAngle>
            <value>73.0</value>
          </takeoffAngle>
          <timeResidual>-1.376574352</timeResidual>
          <timeWeight>0.0</timeWeight>
        </arrival>
        <arrival publicID="smi:org.gfz-potsdam.de/geofon/Pick/20200326153452.374793.258048_NLL.20200416191747.121847.63133">
          <pickID>smi:org.gfz-potsdam.de/geofon/Pick/20200326153452.374793.258048</pickID>
          <phase>S</phase>
          <timeCorrection>0.0</timeCorrection>
          <azimuth>99.15558363</azimuth>
          <distance>0.795902985</distance>
          <takeoffAngle>
            <value>72.9</value>
          </takeoffAngle>
          <timeResidual>-0.9993327423</timeResidual>
          <timeWeight>0.0</timeWeight>
        </arrival>
        <arrival publicID="smi:org.gfz-potsdam.de/geofon/Pick/20200326153452.374822.258049_NLL.20200416191747.121847.63133">
          <pickID>smi:org.gfz-potsdam.de/geofon/Pick/20200326153452.374822.258049</pickID>
          <phase>P</phase>
          <timeCorrection>0.0</timeCorrection>
          <azimuth>99.15558363</azimuth>
          <distance>0.795902985</distance>
          <takeoffAngle>
            <value>72.9</value>
          </takeoffAngle>
          <timeResidual>-0.3056322504</timeResidual>
          <timeWeight>1.038956435</timeWeight>
        </arrival>
        <arrival publicID="smi:org.gfz-potsdam.de/geofon/Pick/20200326153452.374853.258050_NLL.20200416191747.121847.63133">
          <pickID>smi:org.gfz-potsdam.de/geofon/Pick/20200326153452.374853.258050</pickID>
          <phase>P</phase>
          <timeCorrection>0.0</timeCorrection>
          <azimuth>127.7193441</azimuth>
          <distance>0.8450255309</distance>
          <takeoffAngle>
            <value>72.6</value>
          </takeoffAngle>
          <timeResidual>-0.2225268106</timeResidual>
          <timeWeight>0.0</timeWeight>
        </arrival>
        <arrival publicID="smi:org.gfz-potsdam.de/geofon/Pick/20200326153452.374882.258051_NLL.20200416191747.121847.63133">
          <pickID>smi:org.gfz-potsdam.de/geofon/Pick/20200326153452.374882.258051</pickID>
          <phase>S</phase>
          <timeCorrection>0.0</timeCorrection>
          <azimuth>127.7193441</azimuth>
          <distance>0.8450255309</distance>
          <takeoffAngle>
            <value>72.6</value>
          </takeoffAngle>
          <timeResidual>-0.7702076776</timeResidual>
          <timeWeight>0.0</timeWeight>
        </arrival>
        <arrival publicID="smi:org.gfz-potsdam.de/geofon/Pick/20200326153452.37491.258052_NLL.20200416191747.121847.63133">
          <pickID>smi:org.gfz-potsdam.de/geofon/Pick/20200326153452.37491.258052</pickID>
          <phase>S</phase>
          <timeCorrection>0.0</timeCorrection>
          <azimuth>24.84120772</azimuth>
          <distance>0.8489155136</distance>
          <takeoffAngle>
            <value>72.6</value>
          </takeoffAngle>
          <timeResidual>-0.8558210881</timeResidual>
          <timeWeight>0.0</timeWeight>
        </arrival>
        <arrival publicID="smi:org.gfz-potsdam.de/geofon/Pick/20200326153452.374937.258053_NLL.20200416191747.121847.63133">
          <pickID>smi:org.gfz-potsdam.de/geofon/Pick/20200326153452.374937.258053</pickID>
          <phase>P</phase>
          <timeCorrection>0.0</timeCorrection>
          <azimuth>24.84120772</azimuth>
          <distance>0.8489155136</distance>
          <takeoffAngle>
            <value>72.6</value>
          </takeoffAngle>
          <timeResidual>-0.408507045</timeResidual>
          <timeWeight>0.0</timeWeight>
        </arrival>
        <arrival publicID="smi:org.gfz-potsdam.de/geofon/Pick/20200326153452.374966.258054_NLL.20200416191747.121847.63133">
          <pickID>smi:org.gfz-potsdam.de/geofon/Pick/20200326153452.374966.258054</pickID>
          <phase>P</phase>
          <timeCorrection>0.0</timeCorrection>
          <azimuth>40.42339688</azimuth>
          <distance>0.8575796309</distance>
          <takeoffAngle>
            <value>72.6</value>
          </takeoffAngle>
          <timeResidual>-0.4430744203</timeResidual>
          <timeWeight>0.0</timeWeight>
        </arrival>
        <arrival publicID="smi:org.gfz-potsdam.de/geofon/Pick/20200326153452.374992.258055_NLL.20200416191747.121847.63133">
          <pickID>smi:org.gfz-potsdam.de/geofon/Pick/20200326153452.374992.258055</pickID>
          <phase>S</phase>
          <timeCorrection>0.0</timeCorrection>
          <azimuth>40.42339688</azimuth>
          <distance>0.8575796309</distance>
          <takeoffAngle>
            <value>72.6</value>
          </takeoffAngle>
          <timeResidual>-1.339750672</timeResidual>
          <timeWeight>0.0</timeWeight>
        </arrival>
        <arrival publicID="smi:org.gfz-potsdam.de/geofon/Pick/20200326153452.375021.258056_NLL.20200416191747.121847.63133">
          <pickID>smi:org.gfz-potsdam.de/geofon/Pick/20200326153452.375021.258056</pickID>
          <phase>P</phase>
          <timeCorrection>0.0</timeCorrection>
          <azimuth>333.9439152</azimuth>
          <distance>0.8829581856</distance>
          <takeoffAngle>
            <value>72.6</value>
          </takeoffAngle>
          <timeResidual>-0.6147695961</timeResidual>
          <timeWeight>0.77737485</timeWeight>
        </arrival>
        <arrival publicID="smi:org.gfz-potsdam.de/geofon/Pick/20200326153452.375053.258057_NLL.20200416191747.121847.63133">
          <pickID>smi:org.gfz-potsdam.de/geofon/Pick/20200326153452.375053.258057</pickID>
          <phase>P</phase>
          <timeCorrection>0.0</timeCorrection>
          <azimuth>130.2699623</azimuth>
          <distance>0.8958143411</distance>
          <takeoffAngle>
            <value>72.6</value>
          </takeoffAngle>
          <timeResidual>-0.2777544617</timeResidual>
          <timeWeight>0.0</timeWeight>
        </arrival>
        <arrival publicID="smi:org.gfz-potsdam.de/geofon/Pick/20200326153452.375082.258058_NLL.20200416191747.121847.63133">
          <pickID>smi:org.gfz-potsdam.de/geofon/Pick/20200326153452.375082.258058</pickID>
          <phase>S</phase>
          <timeCorrection>0.0</timeCorrection>
          <azimuth>130.2699623</azimuth>
          <distance>0.8958143411</distance>
          <takeoffAngle>
            <value>72.6</value>
          </takeoffAngle>
          <timeResidual>-1.119229558</timeResidual>
          <timeWeight>0.0</timeWeight>
        </arrival>
        <arrival publicID="smi:org.gfz-potsdam.de/geofon/Pick/20200326153452.375112.258059_NLL.20200416191747.121847.63133">
          <pickID>smi:org.gfz-potsdam.de/geofon/Pick/20200326153452.375112.258059</pickID>
          <phase>P</phase>
          <timeCorrection>0.0</timeCorrection>
          <azimuth>116.2014491</azimuth>
          <distance>0.9066051249</distance>
          <takeoffAngle>
            <value>72.6</value>
          </takeoffAngle>
          <timeResidual>-0.5461139976</timeResidual>
          <timeWeight>0.0</timeWeight>
        </arrival>
        <arrival publicID="smi:org.gfz-potsdam.de/geofon/Pick/20200326153452.375144.258060_NLL.20200416191747.121847.63133">
          <pickID>smi:org.gfz-potsdam.de/geofon/Pick/20200326153452.375144.258060</pickID>
          <phase>P</phase>
          <timeCorrection>0.0</timeCorrection>
          <azimuth>113.1360964</azimuth>
          <distance>0.9192475195</distance>
          <takeoffAngle>
            <value>72.6</value>
          </takeoffAngle>
          <timeResidual>-0.4612135302</timeResidual>
          <timeWeight>0.0</timeWeight>
        </arrival>
        <arrival publicID="smi:org.gfz-potsdam.de/geofon/Pick/20200326153452.375175.258061_NLL.20200416191747.121847.63133">
          <pickID>smi:org.gfz-potsdam.de/geofon/Pick/20200326153452.375175.258061</pickID>
          <phase>S</phase>
          <timeCorrection>0.0</timeCorrection>
          <azimuth>113.1360964</azimuth>
          <distance>0.9192475195</distance>
          <takeoffAngle>
            <value>72.6</value>
          </takeoffAngle>
          <timeResidual>-1.820313004</timeResidual>
          <timeWeight>0.0</timeWeight>
        </arrival>
        <arrival publicID="smi:org.gfz-potsdam.de/geofon/Pick/20200326153452.375208.258062_NLL.20200416191747.121847.63133">
          <pickID>smi:org.gfz-potsdam.de/geofon/Pick/20200326153452.375208.258062</pickID>
          <phase>P</phase>
          <timeCorrection>0.0</timeCorrection>
          <azimuth>120.7530369</azimuth>
          <distance>0.9621617767</distance>
          <takeoffAngle>
            <value>72.6</value>
          </takeoffAngle>
          <timeResidual>-0.5108871401</timeResidual>
          <timeWeight>0.0</timeWeight>
        </arrival>
        <arrival publicID="smi:org.gfz-potsdam.de/geofon/Pick/20200326153452.37524.258063_NLL.20200416191747.121847.63133">
          <pickID>smi:org.gfz-potsdam.de/geofon/Pick/20200326153452.37524.258063</pickID>
          <phase>S</phase>
          <timeCorrection>0.0</timeCorrection>
          <azimuth>141.616559</azimuth>
          <distance>1.001110056</distance>
          <takeoffAngle>
            <value>72.6</value>
          </takeoffAngle>
          <timeResidual>-1.304223764</timeResidual>
          <timeWeight>0.0</timeWeight>
        </arrival>
        <arrival publicID="smi:org.gfz-potsdam.de/geofon/Pick/20200326153452.375353.258064_NLL.20200416191747.121847.63133">
          <pickID>smi:org.gfz-potsdam.de/geofon/Pick/20200326153452.375353.258064</pickID>
          <phase>P</phase>
          <timeCorrection>0.0</timeCorrection>
          <azimuth>141.616559</azimuth>
          <distance>1.001110056</distance>
          <takeoffAngle>
            <value>72.6</value>
          </takeoffAngle>
          <timeResidual>-0.6270671613</timeResidual>
          <timeWeight>0.0</timeWeight>
        </arrival>
        <arrival publicID="smi:org.gfz-potsdam.de/geofon/Pick/20200326153452.375392.258065_NLL.20200416191747.121847.63133">
          <pickID>smi:org.gfz-potsdam.de/geofon/Pick/20200326153452.375392.258065</pickID>
          <phase>P</phase>
          <timeCorrection>0.0</timeCorrection>
          <azimuth>329.0670341</azimuth>
          <distance>1.042249921</distance>
          <takeoffAngle>
            <value>72.6</value>
          </takeoffAngle>
          <timeResidual>-0.7213805782</timeResidual>
          <timeWeight>0.694761175</timeWeight>
        </arrival>
        <arrival publicID="smi:org.gfz-potsdam.de/geofon/Pick/20200326153452.375438.258066_NLL.20200416191747.121847.63133">
          <pickID>smi:org.gfz-potsdam.de/geofon/Pick/20200326153452.375438.258066</pickID>
          <phase>S</phase>
          <timeCorrection>0.0</timeCorrection>
          <azimuth>270.4008298</azimuth>
          <distance>1.133449543</distance>
          <takeoffAngle>
            <value>72.6</value>
          </takeoffAngle>
          <timeResidual>-1.379456312</timeResidual>
          <timeWeight>0.0</timeWeight>
        </arrival>
        <arrival publicID="smi:org.gfz-potsdam.de/geofon/Pick/20200326153452.375476.258067_NLL.20200416191747.121847.63133">
          <pickID>smi:org.gfz-potsdam.de/geofon/Pick/20200326153452.375476.258067</pickID>
          <phase>S</phase>
          <timeCorrection>0.0</timeCorrection>
          <azimuth>221.5496962</azimuth>
          <distance>1.216620085</distance>
          <takeoffAngle>
            <value>72.6</value>
          </takeoffAngle>
          <timeResidual>-1.115589696</timeResidual>
          <timeWeight>0.0</timeWeight>
        </arrival>
        <arrival publicID="smi:org.gfz-potsdam.de/geofon/Pick/20200326153452.375506.258068_NLL.20200416191747.121847.63133">
          <pickID>smi:org.gfz-potsdam.de/geofon/Pick/20200326153452.375506.258068</pickID>
          <phase>S</phase>
          <timeCorrection>0.0</timeCorrection>
          <azimuth>129.6134587</azimuth>
          <distance>1.272655498</distance>
          <takeoffAngle>
            <value>72.6</value>
          </takeoffAngle>
          <timeResidual>-2.065633832</timeResidual>
          <timeWeight>0.0</timeWeight>
        </arrival>
        <arrival publicID="smi:org.gfz-potsdam.de/geofon/Pick/20200326153452.375539.258069_NLL.20200416191747.121847.63133">
          <pickID>smi:org.gfz-potsdam.de/geofon/Pick/20200326153452.375539.258069</pickID>
          <phase>P</phase>
          <timeCorrection>0.0</timeCorrection>
          <azimuth>129.6134587</azimuth>
          <distance>1.272655498</distance>
          <takeoffAngle>
            <value>72.6</value>
          </takeoffAngle>
          <timeResidual>-0.8522405957</timeResidual>
          <timeWeight>0.5838719619</timeWeight>
        </arrival>
        <arrival publicID="smi:org.gfz-potsdam.de/geofon/Pick/20200326153452.375571.258070_NLL.20200416191747.121847.63133">
          <pickID>smi:org.gfz-potsdam.de/geofon/Pick/20200326153452.375571.258070</pickID>
          <phase>P</phase>
          <timeCorrection>0.0</timeCorrection>
          <azimuth>6.919212626</azimuth>
          <distance>1.339304998</distance>
          <takeoffAngle>
            <value>72.6</value>
          </takeoffAngle>
          <timeResidual>-0.8981974982</timeResidual>
          <timeWeight>0.0</timeWeight>
        </arrival>
        <arrival publicID="smi:org.gfz-potsdam.de/geofon/Pick/20200326153452.375605.258071_NLL.20200416191747.121847.63133">
          <pickID>smi:org.gfz-potsdam.de/geofon/Pick/20200326153452.375605.258071</pickID>
          <phase>P</phase>
          <timeCorrection>0.0</timeCorrection>
          <azimuth>71.74084968</azimuth>
          <distance>1.344712016</distance>
          <takeoffAngle>
            <value>72.6</value>
          </takeoffAngle>
          <timeResidual>-0.9617221697</timeResidual>
          <timeWeight>0.0</timeWeight>
        </arrival>
        <arrival publicID="smi:org.gfz-potsdam.de/geofon/Pick/20200326153452.375635.258072_NLL.20200416191747.121847.63133">
          <pickID>smi:org.gfz-potsdam.de/geofon/Pick/20200326153452.375635.258072</pickID>
          <phase>S</phase>
          <timeCorrection>0.0</timeCorrection>
          <azimuth>71.74084968</azimuth>
          <distance>1.344712016</distance>
          <takeoffAngle>
            <value>72.6</value>
          </takeoffAngle>
          <timeResidual>-2.42724481</timeResidual>
          <timeWeight>0.0</timeWeight>
        </arrival>
        <arrival publicID="smi:org.gfz-potsdam.de/geofon/Pick/20200326153452.375671.258073_NLL.20200416191747.121847.63133">
          <pickID>smi:org.gfz-potsdam.de/geofon/Pick/20200326153452.375671.258073</pickID>
          <phase>S</phase>
          <timeCorrection>0.0</timeCorrection>
          <azimuth>165.206219</azimuth>
          <distance>1.376788589</distance>
          <takeoffAngle>
            <value>72.6</value>
          </takeoffAngle>
          <timeResidual>-2.890932973</timeResidual>
          <timeWeight>0.0</timeWeight>
        </arrival>
        <arrival publicID="smi:org.gfz-potsdam.de/geofon/Pick/20200326153452.375699.258074_NLL.20200416191747.121847.63133">
          <pickID>smi:org.gfz-potsdam.de/geofon/Pick/20200326153452.375699.258074</pickID>
          <phase>P</phase>
          <timeCorrection>0.0</timeCorrection>
          <azimuth>165.206219</azimuth>
          <distance>1.376788589</distance>
          <takeoffAngle>
            <value>72.6</value>
          </takeoffAngle>
          <timeResidual>-1.526492514</timeResidual>
          <timeWeight>0.0</timeWeight>
        </arrival>
        <arrival publicID="smi:org.gfz-potsdam.de/geofon/Pick/20200326153452.375731.258075_NLL.20200416191747.121847.63133">
          <pickID>smi:org.gfz-potsdam.de/geofon/Pick/20200326153452.375731.258075</pickID>
          <phase>P</phase>
          <timeCorrection>0.0</timeCorrection>
          <azimuth>78.75087683</azimuth>
          <distance>1.5598244</distance>
          <takeoffAngle>
            <value>46.0</value>
          </takeoffAngle>
          <timeResidual>-0.8652377407</timeResidual>
          <timeWeight>0.0</timeWeight>
        </arrival>
        <arrival publicID="smi:org.gfz-potsdam.de/geofon/Pick/20200326153452.375766.258076_NLL.20200416191747.121847.63133">
          <pickID>smi:org.gfz-potsdam.de/geofon/Pick/20200326153452.375766.258076</pickID>
          <phase>S</phase>
          <timeCorrection>0.0</timeCorrection>
          <azimuth>106.0916839</azimuth>
          <distance>1.793804537</distance>
          <takeoffAngle>
            <value>46.0</value>
          </takeoffAngle>
          <timeResidual>-1.34352653</timeResidual>
          <timeWeight>0.0</timeWeight>
        </arrival>
        <arrival publicID="smi:org.gfz-potsdam.de/geofon/Pick/20200326153452.375795.258077_NLL.20200416191747.121847.63133">
          <pickID>smi:org.gfz-potsdam.de/geofon/Pick/20200326153452.375795.258077</pickID>
          <phase>P</phase>
          <timeCorrection>0.0</timeCorrection>
          <azimuth>106.0916839</azimuth>
          <distance>1.793804537</distance>
          <takeoffAngle>
            <value>46.0</value>
          </takeoffAngle>
          <timeResidual>-0.2590232722</timeResidual>
          <timeWeight>0.9840506275</timeWeight>
        </arrival>
        <arrival publicID="smi:org.gfz-potsdam.de/geofon/Pick/20200326153452.375825.258078_NLL.20200416191747.121847.63133">
          <pickID>smi:org.gfz-potsdam.de/geofon/Pick/20200326153452.375825.258078</pickID>
          <phase>P</phase>
          <timeCorrection>0.0</timeCorrection>
          <azimuth>81.47485626</azimuth>
          <distance>1.936217947</distance>
          <takeoffAngle>
            <value>46.0</value>
          </takeoffAngle>
          <timeResidual>0.2035404266</timeResidual>
          <timeWeight>0.9168770792</timeWeight>
        </arrival>
        <arrival publicID="smi:org.gfz-potsdam.de/geofon/Pick/20200326153452.375855.258079_NLL.20200416191747.121847.63133">
          <pickID>smi:org.gfz-potsdam.de/geofon/Pick/20200326153452.375855.258079</pickID>
          <phase>S</phase>
          <timeCorrection>0.0</timeCorrection>
          <azimuth>81.47485626</azimuth>
          <distance>1.936217947</distance>
          <takeoffAngle>
            <value>46.0</value>
          </takeoffAngle>
          <timeResidual>-0.9718344297</timeResidual>
          <timeWeight>0.0</timeWeight>
        </arrival>
        <arrival publicID="smi:org.gfz-potsdam.de/geofon/Pick/20200326153452.375887.258080_NLL.20200416191747.121847.63133">
          <pickID>smi:org.gfz-potsdam.de/geofon/Pick/20200326153452.375887.258080</pickID>
          <phase>S</phase>
          <timeCorrection>0.0</timeCorrection>
          <azimuth>272.7989453</azimuth>
          <distance>2.091886725</distance>
          <takeoffAngle>
            <value>45.1</value>
          </takeoffAngle>
          <timeResidual>1.710511385</timeResidual>
          <timeWeight>0.0</timeWeight>
        </arrival>
        <arrival publicID="smi:org.gfz-potsdam.de/geofon/Pick/20200326153452.375945.258082_NLL.20200416191747.121847.63133">
          <pickID>smi:org.gfz-potsdam.de/geofon/Pick/20200326153452.375945.258082</pickID>
          <phase>S</phase>
          <timeCorrection>0.0</timeCorrection>
          <azimuth>63.21588255</azimuth>
          <distance>2.076665553</distance>
          <takeoffAngle>
            <value>45.1</value>
          </takeoffAngle>
          <timeResidual>0.4723391272</timeResidual>
          <timeWeight>0.6145667982</timeWeight>
        </arrival>
        <arrival publicID="smi:org.gfz-potsdam.de/geofon/Pick/20200326153452.375973.258083_NLL.20200416191747.121847.63133">
          <pickID>smi:org.gfz-potsdam.de/geofon/Pick/20200326153452.375973.258083</pickID>
          <phase>P</phase>
          <timeCorrection>0.0</timeCorrection>
          <azimuth>63.21588255</azimuth>
          <distance>2.076665553</distance>
          <takeoffAngle>
            <value>45.1</value>
          </takeoffAngle>
          <timeResidual>0.07408531285</timeResidual>
          <timeWeight>0.8291756997</timeWeight>
        </arrival>
        <arrival publicID="smi:org.gfz-potsdam.de/geofon/Pick/20200326153452.37601.258084_NLL.20200416191747.121847.63133">
          <pickID>smi:org.gfz-potsdam.de/geofon/Pick/20200326153452.37601.258084</pickID>
          <phase>P</phase>
          <timeCorrection>0.0</timeCorrection>
          <azimuth>172.1959522</azimuth>
          <distance>2.395360102</distance>
          <takeoffAngle>
            <value>45.1</value>
          </takeoffAngle>
          <timeResidual>-0.03824017591</timeResidual>
          <timeWeight>0.7673029685</timeWeight>
        </arrival>
        <arrival publicID="smi:org.gfz-potsdam.de/geofon/Pick/20200326154509.418524.259129_NLL.20200416191747.121847.63133">
          <pickID>smi:org.gfz-potsdam.de/geofon/Pick/20200326154509.418524.259129</pickID>
          <phase>S</phase>
          <timeCorrection>0.0</timeCorrection>
          <azimuth>130.3717009</azimuth>
          <distance>0.2303010636</distance>
          <takeoffAngle>
            <value>82.6</value>
          </takeoffAngle>
          <timeResidual>0.04185200322</timeResidual>
          <timeWeight>1.303023846</timeWeight>
        </arrival>
        <arrival publicID="smi:org.gfz-potsdam.de/geofon/Pick/20200326154509.418628.259130_NLL.20200416191747.121847.63133">
          <pickID>smi:org.gfz-potsdam.de/geofon/Pick/20200326154509.418628.259130</pickID>
          <phase>S</phase>
          <timeCorrection>0.0</timeCorrection>
          <azimuth>110.7805778</azimuth>
          <distance>0.3500816135</distance>
          <takeoffAngle>
            <value>82.0</value>
          </takeoffAngle>
          <timeResidual>-0.06774223431</timeResidual>
          <timeWeight>1.111128779</timeWeight>
        </arrival>
        <arrival publicID="smi:org.gfz-potsdam.de/geofon/Pick/20200326154509.41868.259131_NLL.20200416191747.121847.63133">
          <pickID>smi:org.gfz-potsdam.de/geofon/Pick/20200326154509.41868.259131</pickID>
          <phase>S</phase>
          <timeCorrection>0.0</timeCorrection>
          <azimuth>154.7843702</azimuth>
          <distance>0.3682233867</distance>
          <takeoffAngle>
            <value>81.9</value>
          </takeoffAngle>
          <timeResidual>-0.5844104289</timeResidual>
          <timeWeight>0.8302923309</timeWeight>
        </arrival>
        <arrival publicID="smi:org.gfz-potsdam.de/geofon/Pick/20200326154509.418812.259132_NLL.20200416191747.121847.63133">
          <pickID>smi:org.gfz-potsdam.de/geofon/Pick/20200326154509.418812.259132</pickID>
          <phase>S</phase>
          <timeCorrection>0.0</timeCorrection>
          <azimuth>108.1462602</azimuth>
          <distance>1.160328911</distance>
          <takeoffAngle>
            <value>72.6</value>
          </takeoffAngle>
          <timeResidual>-1.795007011</timeResidual>
          <timeWeight>0.0</timeWeight>
        </arrival>
        <arrival publicID="smi:org.gfz-potsdam.de/geofon/Pick/20200326154509.418863.259133_NLL.20200416191747.121847.63133">
          <pickID>smi:org.gfz-potsdam.de/geofon/Pick/20200326154509.418863.259133</pickID>
          <phase>S</phase>
          <timeCorrection>0.0</timeCorrection>
          <azimuth>78.75087683</azimuth>
          <distance>1.5598244</distance>
          <takeoffAngle>
            <value>46.0</value>
          </takeoffAngle>
          <timeResidual>-2.958849435</timeResidual>
          <timeWeight>0.0</timeWeight>
        </arrival>
        <arrival publicID="smi:org.gfz-potsdam.de/geofon/Pick/20200326154509.418911.259134_NLL.20200416191747.121847.63133">
          <pickID>smi:org.gfz-potsdam.de/geofon/Pick/20200326154509.418911.259134</pickID>
          <phase>P</phase>
          <timeCorrection>0.0</timeCorrection>
          <azimuth>272.7989453</azimuth>
          <distance>2.091886725</distance>
          <takeoffAngle>
            <value>45.1</value>
          </takeoffAngle>
          <timeResidual>0.175074416</timeResidual>
          <timeWeight>0.8605199862</timeWeight>
        </arrival>
        <arrival publicID="smi:org.gfz-potsdam.de/geofon/Pick/20200416191248.529211.62721_NLL.20200416191747.121847.63133">
          <pickID>smi:org.gfz-potsdam.de/geofon/Pick/20200416191248.529211.62721</pickID>
          <phase>S</phase>
          <timeCorrection>0.0</timeCorrection>
          <azimuth>288.9808605</azimuth>
          <distance>0.224160181</distance>
          <takeoffAngle>
            <value>82.6</value>
          </takeoffAngle>
          <timeResidual>0.06715659248</timeResidual>
          <timeWeight>1.290225143</timeWeight>
        </arrival>
        <arrival publicID="smi:org.gfz-potsdam.de/geofon/Pick/20200416191248.529382.62722_NLL.20200416191747.121847.63133">
          <pickID>smi:org.gfz-potsdam.de/geofon/Pick/20200416191248.529382.62722</pickID>
          <phase>P</phase>
          <timeCorrection>0.0</timeCorrection>
          <azimuth>288.9808605</azimuth>
          <distance>0.224160181</distance>
          <takeoffAngle>
            <value>82.6</value>
          </takeoffAngle>
          <timeResidual>0.3967197695</timeResidual>
          <timeWeight>0.8668669456</timeWeight>
        </arrival>
      </origin>
      <magnitude publicID="smi:org.gfz-potsdam.de/geofon/NLL.20200416191747.121847.63133/netMag/ML(TexNet)">
        <mag>
          <value>4.911862396</value>
          <uncertainty>0.1828989451</uncertainty>
        </mag>
        <type>ML(TexNet)</type>
        <originID>smi:org.gfz-potsdam.de/geofon/NLL.20200416191747.121847.63133</originID>
        <methodID>smi:org.gfz-potsdam.de/geofon/Trimmed_mean,_overall_err_&lt;_0.2</methodID>
        <stationCount>18</stationCount>
        <creationInfo>
          <agencyID>TXNet</agencyID>
          <author>scmag@sc3primary.beg.utexas.edu</author>
          <creationTime>2020-10-15T21:12:05.235975Z</creationTime>
        </creationInfo>
      </magnitude>
      <pick publicID="smi:org.gfz-potsdam.de/geofon/Pick/20200326152204.026602.256215">
        <time>
          <value>2020-03-26T15:16:35.395864Z</value>
          <lowerUncertainty>0.200000003</lowerUncertainty>
          <upperUncertainty>0.200000003</upperUncertainty>
        </time>
        <waveformID networkCode="TX" stationCode="PB07" locationCode="00" channelCode="HHZ"></waveformID>
        <phaseHint>P</phaseHint>
        <evaluationMode>manual</evaluationMode>
        <creationInfo>
          <agencyID>TXNet</agencyID>
          <author>Tricia</author>
          <creationTime>2020-03-26T15:22:04.026725Z</creationTime>
        </creationInfo>
      </pick>
      <pick publicID="smi:org.gfz-potsdam.de/geofon/Pick/20200326152204.026802.256217">
        <time>
          <value>2020-03-26T15:16:35.433116Z</value>
          <lowerUncertainty>0.3000000119</lowerUncertainty>
          <upperUncertainty>0.3000000119</upperUncertainty>
        </time>
        <waveformID networkCode="TX" stationCode="PECS" locationCode="00" channelCode="HHZ"></waveformID>
        <phaseHint>P</phaseHint>
        <evaluationMode>manual</evaluationMode>
        <creationInfo>
          <agencyID>TXNet</agencyID>
          <author>Tricia</author>
          <creationTime>2020-03-26T15:22:04.026824Z</creationTime>
        </creationInfo>
      </pick>
      <pick publicID="smi:org.gfz-potsdam.de/geofon/Pick/20200326152204.026844.256218">
        <time>
          <value>2020-03-26T15:16:37.755157Z</value>
          <lowerUncertainty>0.3000000119</lowerUncertainty>
          <upperUncertainty>0.3000000119</upperUncertainty>
        </time>
        <waveformID networkCode="TX" stationCode="PB10" locationCode="00" channelCode="HHZ"></waveformID>
        <phaseHint>P</phaseHint>
        <evaluationMode>manual</evaluationMode>
        <creationInfo>
          <agencyID>TXNet</agencyID>
          <author>Tricia</author>
          <creationTime>2020-03-26T15:22:04.026872Z</creationTime>
        </creationInfo>
      </pick>
      <pick publicID="smi:org.gfz-potsdam.de/geofon/Pick/20200326152204.026887.256219">
        <time>
          <value>2020-03-26T15:16:38.984473Z</value>
          <lowerUncertainty>0.400000006</lowerUncertainty>
          <upperUncertainty>0.400000006</upperUncertainty>
        </time>
        <waveformID networkCode="TX" stationCode="PCOS" locationCode="00" channelCode="HHZ"></waveformID>
        <phaseHint>P</phaseHint>
        <evaluationMode>manual</evaluationMode>
        <creationInfo>
          <agencyID>TXNet</agencyID>
          <author>Tricia</author>
          <creationTime>2020-03-26T15:22:04.026907Z</creationTime>
        </creationInfo>
      </pick>
      <pick publicID="smi:org.gfz-potsdam.de/geofon/Pick/20200326152204.026919.256220">
        <time>
          <value>2020-03-26T15:16:47.080576Z</value>
          <lowerUncertainty>0.400000006</lowerUncertainty>
          <upperUncertainty>0.400000006</upperUncertainty>
        </time>
        <waveformID networkCode="TX" stationCode="PCOS" locationCode="00" channelCode="HHE"></waveformID>
        <phaseHint>S</phaseHint>
        <evaluationMode>manual</evaluationMode>
        <creationInfo>
          <agencyID>TXNet</agencyID>
          <author>Tricia</author>
          <creationTime>2020-03-26T15:22:04.026937Z</creationTime>
        </creationInfo>
      </pick>
      <pick publicID="smi:org.gfz-potsdam.de/geofon/Pick/20200326152204.026952.256221">
        <time>
          <value>2020-03-26T15:16:47.813199Z</value>
          <lowerUncertainty>0.400000006</lowerUncertainty>
          <upperUncertainty>0.400000006</upperUncertainty>
        </time>
        <waveformID networkCode="TX" stationCode="DG03" locationCode="00" channelCode="CH2"></waveformID>
        <phaseHint>S</phaseHint>
        <evaluationMode>manual</evaluationMode>
        <creationInfo>
          <agencyID>TXNet</agencyID>
          <author>Tricia</author>
          <creationTime>2020-03-26T15:22:04.026971Z</creationTime>
        </creationInfo>
      </pick>
      <pick publicID="smi:org.gfz-potsdam.de/geofon/Pick/20200326152204.026985.256222">
        <time>
          <value>2020-03-26T15:16:39.605340Z</value>
          <lowerUncertainty>0.3000000119</lowerUncertainty>
          <upperUncertainty>0.3000000119</upperUncertainty>
        </time>
        <waveformID networkCode="TX" stationCode="DG03" locationCode="00" channelCode="CHZ"></waveformID>
        <phaseHint>P</phaseHint>
        <evaluationMode>manual</evaluationMode>
        <creationInfo>
          <agencyID>TXNet</agencyID>
          <author>Tricia</author>
          <creationTime>2020-03-26T15:22:04.027003Z</creationTime>
        </creationInfo>
      </pick>
      <pick publicID="smi:org.gfz-potsdam.de/geofon/Pick/20200326152204.027024.256223">
        <time>
          <value>2020-03-26T15:16:37.854496Z</value>
          <lowerUncertainty>0.3000000119</lowerUncertainty>
          <upperUncertainty>0.3000000119</upperUncertainty>
        </time>
        <waveformID networkCode="TX" stationCode="PB12" locationCode="00" channelCode="HHZ"></waveformID>
        <phaseHint>P</phaseHint>
        <evaluationMode>manual</evaluationMode>
        <creationInfo>
          <agencyID>TXNet</agencyID>
          <author>Tricia</author>
          <creationTime>2020-03-26T15:22:04.027044Z</creationTime>
        </creationInfo>
      </pick>
      <pick publicID="smi:org.gfz-potsdam.de/geofon/Pick/20200326152204.027057.256224">
        <time>
          <value>2020-03-26T15:16:45.180724Z</value>
          <lowerUncertainty>0.400000006</lowerUncertainty>
          <upperUncertainty>0.400000006</upperUncertainty>
        </time>
        <waveformID networkCode="TX" stationCode="PB12" locationCode="00" channelCode="HHE"></waveformID>
        <phaseHint>S</phaseHint>
        <evaluationMode>manual</evaluationMode>
        <creationInfo>
          <agencyID>TXNet</agencyID>
          <author>Tricia</author>
          <creationTime>2020-03-26T15:22:04.027133Z</creationTime>
        </creationInfo>
      </pick>
      <pick publicID="smi:org.gfz-potsdam.de/geofon/Pick/20200326152204.027216.256226">
        <time>
          <value>2020-03-26T15:16:36.662432Z</value>
          <lowerUncertainty>0.5</lowerUncertainty>
          <upperUncertainty>0.5</upperUncertainty>
        </time>
        <waveformID networkCode="TX" stationCode="PB11" locationCode="00" channelCode="HH1"></waveformID>
        <phaseHint>S</phaseHint>
        <evaluationMode>manual</evaluationMode>
        <creationInfo>
          <agencyID>TXNet</agencyID>
          <author>Tricia</author>
          <creationTime>2020-03-26T15:22:04.027235Z</creationTime>
        </creationInfo>
      </pick>
      <pick publicID="smi:org.gfz-potsdam.de/geofon/Pick/20200326152204.027277.256228">
        <time>
          <value>2020-03-26T15:16:42.523414Z</value>
          <lowerUncertainty>0.400000006</lowerUncertainty>
          <upperUncertainty>0.400000006</upperUncertainty>
        </time>
        <waveformID networkCode="TX" stationCode="DG01" locationCode="00" channelCode="CHZ"></waveformID>
        <phaseHint>P</phaseHint>
        <evaluationMode>manual</evaluationMode>
        <creationInfo>
          <agencyID>TXNet</agencyID>
          <author>Tricia</author>
          <creationTime>2020-03-26T15:22:04.027296Z</creationTime>
        </creationInfo>
      </pick>
      <pick publicID="smi:org.gfz-potsdam.de/geofon/Pick/20200326152204.02731.256229">
        <time>
          <value>2020-03-26T15:16:42.734509Z</value>
          <lowerUncertainty>0.5</lowerUncertainty>
          <upperUncertainty>0.5</upperUncertainty>
        </time>
        <waveformID networkCode="TX" stationCode="PB30" locationCode="00" channelCode="HHZ"></waveformID>
        <phaseHint>P</phaseHint>
        <evaluationMode>manual</evaluationMode>
        <creationInfo>
          <agencyID>TXNet</agencyID>
          <author>Tricia</author>
          <creationTime>2020-03-26T15:22:04.027332Z</creationTime>
        </creationInfo>
      </pick>
      <pick publicID="smi:org.gfz-potsdam.de/geofon/Pick/20200326152204.027345.256230">
        <time>
          <value>2020-03-26T15:16:53.537591Z</value>
          <lowerUncertainty>0.5</lowerUncertainty>
          <upperUncertainty>0.5</upperUncertainty>
        </time>
        <waveformID networkCode="TX" stationCode="PB30" locationCode="00" channelCode="HH1"></waveformID>
        <phaseHint>S</phaseHint>
        <evaluationMode>manual</evaluationMode>
        <creationInfo>
          <agencyID>TXNet</agencyID>
          <author>Tricia</author>
          <creationTime>2020-03-26T15:22:04.027364Z</creationTime>
        </creationInfo>
      </pick>
      <pick publicID="smi:org.gfz-potsdam.de/geofon/Pick/20200326152313.186867.256298">
        <time>
          <value>2020-03-26T15:16:36.215408Z</value>
          <lowerUncertainty>0.5</lowerUncertainty>
          <upperUncertainty>0.5</upperUncertainty>
        </time>
        <waveformID networkCode="TX" stationCode="PB28" locationCode="00" channelCode="HHZ"></waveformID>
        <phaseHint>P</phaseHint>
        <evaluationMode>manual</evaluationMode>
        <creationInfo>
          <agencyID>TXNet</agencyID>
          <author>Tricia</author>
          <creationTime>2020-03-26T15:23:13.186987Z</creationTime>
        </creationInfo>
      </pick>
      <pick publicID="smi:org.gfz-potsdam.de/geofon/Pick/20200326152313.187024.256299">
        <time>
          <value>2020-03-26T15:16:41.492775Z</value>
          <lowerUncertainty>0.400000006</lowerUncertainty>
          <upperUncertainty>0.400000006</upperUncertainty>
        </time>
        <waveformID networkCode="TX" stationCode="PB28" locationCode="00" channelCode="HH1"></waveformID>
        <phaseHint>S</phaseHint>
        <evaluationMode>manual</evaluationMode>
        <creationInfo>
          <agencyID>TXNet</agencyID>
          <author>Tricia</author>
          <creationTime>2020-03-26T15:23:13.187047Z</creationTime>
        </creationInfo>
      </pick>
      <pick publicID="smi:org.gfz-potsdam.de/geofon/Pick/20200326152313.187063.256300">
        <time>
          <value>2020-03-26T15:16:41.902547Z</value>
          <lowerUncertainty>0.400000006</lowerUncertainty>
          <upperUncertainty>0.400000006</upperUncertainty>
        </time>
        <waveformID networkCode="TX" stationCode="PB29" locationCode="00" channelCode="HH1"></waveformID>
        <phaseHint>S</phaseHint>
        <evaluationMode>manual</evaluationMode>
        <creationInfo>
          <agencyID>TXNet</agencyID>
          <author>Tricia</author>
          <creationTime>2020-03-26T15:23:13.187083Z</creationTime>
        </creationInfo>
      </pick>
      <pick publicID="smi:org.gfz-potsdam.de/geofon/Pick/20200326152313.187094.256301">
        <time>
          <value>2020-03-26T15:16:36.476172Z</value>
          <lowerUncertainty>0.400000006</lowerUncertainty>
          <upperUncertainty>0.400000006</upperUncertainty>
        </time>
        <waveformID networkCode="TX" stationCode="PB29" locationCode="00" channelCode="HHZ"></waveformID>
        <phaseHint>P</phaseHint>
        <evaluationMode>manual</evaluationMode>
        <creationInfo>
          <agencyID>TXNet</agencyID>
          <author>Tricia</author>
          <creationTime>2020-03-26T15:23:13.187113Z</creationTime>
        </creationInfo>
      </pick>
      <pick publicID="smi:org.gfz-potsdam.de/geofon/Pick/20200326152313.187125.256302">
        <time>
          <value>2020-03-26T15:16:36.650014Z</value>
          <lowerUncertainty>0.3000000119</lowerUncertainty>
          <upperUncertainty>0.3000000119</upperUncertainty>
        </time>
        <waveformID networkCode="GM" stationCode="NMP02" locationCode="01" channelCode="HHZ"></waveformID>
        <phaseHint>P</phaseHint>
        <evaluationMode>manual</evaluationMode>
        <creationInfo>
          <agencyID>TXNet</agencyID>
          <author>Tricia</author>
          <creationTime>2020-03-26T15:23:13.187143Z</creationTime>
        </creationInfo>
      </pick>
      <pick publicID="smi:org.gfz-potsdam.de/geofon/Pick/20200326152313.187152.256303">
        <time>
          <value>2020-03-26T15:16:43.156698Z</value>
          <lowerUncertainty>0.400000006</lowerUncertainty>
          <upperUncertainty>0.400000006</upperUncertainty>
        </time>
        <waveformID networkCode="GM" stationCode="NMP02" locationCode="01" channelCode="HHE"></waveformID>
        <phaseHint>S</phaseHint>
        <evaluationMode>manual</evaluationMode>
        <creationInfo>
          <agencyID>TXNet</agencyID>
          <author>Tricia</author>
          <creationTime>2020-03-26T15:23:13.187169Z</creationTime>
        </creationInfo>
      </pick>
      <pick publicID="smi:org.gfz-potsdam.de/geofon/Pick/20200326152313.187183.256304">
        <time>
          <value>2020-03-26T15:16:45.292480Z</value>
          <lowerUncertainty>0.400000006</lowerUncertainty>
          <upperUncertainty>0.400000006</upperUncertainty>
        </time>
        <waveformID networkCode="TX" stationCode="PB10" locationCode="00" channelCode="HH1"></waveformID>
        <phaseHint>S</phaseHint>
        <evaluationMode>manual</evaluationMode>
        <creationInfo>
          <agencyID>TXNet</agencyID>
          <author>Tricia</author>
          <creationTime>2020-03-26T15:23:13.187202Z</creationTime>
        </creationInfo>
      </pick>
      <pick publicID="smi:org.gfz-potsdam.de/geofon/Pick/20200326152313.187217.256305">
        <time>
          <value>2020-03-26T15:16:46.335536Z</value>
          <lowerUncertainty>0.400000006</lowerUncertainty>
          <upperUncertainty>0.400000006</upperUncertainty>
        </time>
        <waveformID networkCode="GM" stationCode="NMP01" locationCode="01" channelCode="HHN"></waveformID>
        <phaseHint>S</phaseHint>
        <evaluationMode>manual</evaluationMode>
        <creationInfo>
          <agencyID>TXNet</agencyID>
          <author>Tricia</author>
          <creationTime>2020-03-26T15:23:13.187235Z</creationTime>
        </creationInfo>
      </pick>
      <pick publicID="smi:org.gfz-potsdam.de/geofon/Pick/20200326152313.187244.256306">
        <time>
          <value>2020-03-26T15:16:38.537449Z</value>
          <lowerUncertainty>0.400000006</lowerUncertainty>
          <upperUncertainty>0.400000006</upperUncertainty>
        </time>
        <waveformID networkCode="GM" stationCode="NMP01" locationCode="01" channelCode="HHZ"></waveformID>
        <phaseHint>P</phaseHint>
        <evaluationMode>manual</evaluationMode>
        <creationInfo>
          <agencyID>TXNet</agencyID>
          <author>Tricia</author>
          <creationTime>2020-03-26T15:23:13.187262Z</creationTime>
        </creationInfo>
      </pick>
      <pick publicID="smi:org.gfz-potsdam.de/geofon/Pick/20200326152313.187274.256307">
        <time>
          <value>2020-03-26T15:16:38.984474Z</value>
          <lowerUncertainty>0.400000006</lowerUncertainty>
          <upperUncertainty>0.400000006</upperUncertainty>
        </time>
        <waveformID networkCode="SC" stationCode="PDB" locationCode="00" channelCode="HHZ"></waveformID>
        <phaseHint>P</phaseHint>
        <evaluationMode>manual</evaluationMode>
        <creationInfo>
          <agencyID>TXNet</agencyID>
          <author>Tricia</author>
          <creationTime>2020-03-26T15:23:13.187293Z</creationTime>
        </creationInfo>
      </pick>
      <pick publicID="smi:org.gfz-potsdam.de/geofon/Pick/20200326152313.187308.256308">
        <time>
          <value>2020-03-26T15:16:39.021726Z</value>
          <lowerUncertainty>0.400000006</lowerUncertainty>
          <upperUncertainty>0.400000006</upperUncertainty>
        </time>
        <waveformID networkCode="SC" stationCode="GDL2" locationCode="00" channelCode="EHZ"></waveformID>
        <phaseHint>P</phaseHint>
        <evaluationMode>manual</evaluationMode>
        <creationInfo>
          <agencyID>TXNet</agencyID>
          <author>Tricia</author>
          <creationTime>2020-03-26T15:23:13.187326Z</creationTime>
        </creationInfo>
      </pick>
      <pick publicID="smi:org.gfz-potsdam.de/geofon/Pick/20200326152358.946492.256577">
        <time>
          <value>2020-03-26T15:16:47.701443Z</value>
          <lowerUncertainty>0.5</lowerUncertainty>
          <upperUncertainty>0.5</upperUncertainty>
        </time>
        <waveformID networkCode="TX" stationCode="PB17" locationCode="00" channelCode="HHZ"></waveformID>
        <phaseHint>P</phaseHint>
        <evaluationMode>manual</evaluationMode>
        <creationInfo>
          <agencyID>TXNet</agencyID>
          <author>Tricia</author>
          <creationTime>2020-03-26T15:23:58.946587Z</creationTime>
        </creationInfo>
      </pick>
      <pick publicID="smi:org.gfz-potsdam.de/geofon/Pick/20200326152358.946643.256578">
        <time>
          <value>2020-03-26T15:16:49.415035Z</value>
          <lowerUncertainty>0.5</lowerUncertainty>
          <upperUncertainty>0.5</upperUncertainty>
        </time>
        <waveformID networkCode="US" stationCode="MNTX" locationCode="00" channelCode="BHZ"></waveformID>
        <phaseHint>P</phaseHint>
        <evaluationMode>manual</evaluationMode>
        <creationInfo>
          <agencyID>TXNet</agencyID>
          <author>Tricia</author>
          <creationTime>2020-03-26T15:23:58.946670Z</creationTime>
        </creationInfo>
      </pick>
      <pick publicID="smi:org.gfz-potsdam.de/geofon/Pick/20200326152358.946685.256579">
        <time>
          <value>2020-03-26T15:16:49.117019Z</value>
          <lowerUncertainty>0.400000006</lowerUncertainty>
          <upperUncertainty>0.400000006</upperUncertainty>
        </time>
        <waveformID networkCode="TX" stationCode="MNHN" locationCode="00" channelCode="HHZ"></waveformID>
        <phaseHint>P</phaseHint>
        <evaluationMode>manual</evaluationMode>
        <creationInfo>
          <agencyID>TXNet</agencyID>
          <author>Tricia</author>
          <creationTime>2020-03-26T15:23:58.946704Z</creationTime>
        </creationInfo>
      </pick>
      <pick publicID="smi:org.gfz-potsdam.de/geofon/Pick/20200326152358.946718.256580">
        <time>
          <value>2020-03-26T15:16:50.594682Z</value>
          <lowerUncertainty>0.6000000238</lowerUncertainty>
          <upperUncertainty>0.6000000238</upperUncertainty>
        </time>
        <waveformID networkCode="TX" stationCode="VHRN" locationCode="00" channelCode="HHZ"></waveformID>
        <phaseHint>P</phaseHint>
        <evaluationMode>manual</evaluationMode>
        <creationInfo>
          <agencyID>TXNet</agencyID>
          <author>Tricia</author>
          <creationTime>2020-03-26T15:23:58.946737Z</creationTime>
        </creationInfo>
      </pick>
      <pick publicID="smi:org.gfz-potsdam.de/geofon/Pick/20200326153452.37426.258036">
        <time>
          <value>2020-03-26T15:16:33.207005Z</value>
          <lowerUncertainty>0.200000003</lowerUncertainty>
          <upperUncertainty>0.200000003</upperUncertainty>
        </time>
        <waveformID networkCode="TX" stationCode="PB11" locationCode="00" channelCode="HHZ"></waveformID>
        <phaseHint>P</phaseHint>
        <polarity>positive</polarity>
        <evaluationMode>manual</evaluationMode>
        <creationInfo>
          <agencyID>TXNet</agencyID>
          <author>Tricia</author>
          <creationTime>2020-03-26T15:34:52.374346Z</creationTime>
        </creationInfo>
      </pick>
      <pick publicID="smi:org.gfz-potsdam.de/geofon/Pick/20200326153452.374363.258037">
        <time>
          <value>2020-03-26T15:16:33.235247Z</value>
          <lowerUncertainty>0.3000000119</lowerUncertainty>
          <upperUncertainty>0.3000000119</upperUncertainty>
        </time>
        <waveformID networkCode="TX" stationCode="PB13" locationCode="00" channelCode="HHZ"></waveformID>
        <phaseHint>P</phaseHint>
        <polarity>positive</polarity>
        <evaluationMode>manual</evaluationMode>
        <creationInfo>
          <agencyID>TXNet</agencyID>
          <author>Tricia</author>
          <creationTime>2020-03-26T15:34:52.374374Z</creationTime>
        </creationInfo>
      </pick>
      <pick publicID="smi:org.gfz-potsdam.de/geofon/Pick/20200326153452.374402.258038">
        <time>
          <value>2020-03-26T15:16:46.571654Z</value>
          <lowerUncertainty>0.3000000119</lowerUncertainty>
          <upperUncertainty>0.3000000119</upperUncertainty>
        </time>
        <waveformID networkCode="SC" stationCode="PDB" locationCode="00" channelCode="HHT"></waveformID>
        <phaseHint>S</phaseHint>
        <evaluationMode>manual</evaluationMode>
        <creationInfo>
          <agencyID>TXNet</agencyID>
          <author>Tricia</author>
          <creationTime>2020-03-26T15:34:52.374452Z</creationTime>
        </creationInfo>
      </pick>
      <pick publicID="smi:org.gfz-potsdam.de/geofon/Pick/20200326153452.374485.258039">
        <time>
          <value>2020-03-26T15:16:48.303056Z</value>
          <lowerUncertainty>0.200000003</lowerUncertainty>
          <upperUncertainty>0.200000003</upperUncertainty>
        </time>
        <waveformID networkCode="4T" stationCode="NM02" locationCode="00" channelCode="HHT"></waveformID>
        <phaseHint>S</phaseHint>
        <evaluationMode>manual</evaluationMode>
        <creationInfo>
          <agencyID>TXNet</agencyID>
          <author>Tricia</author>
          <creationTime>2020-03-26T15:34:52.374510Z</creationTime>
        </creationInfo>
      </pick>
      <pick publicID="smi:org.gfz-potsdam.de/geofon/Pick/20200326153452.37452.258040">
        <time>
          <value>2020-03-26T15:16:39.384466Z</value>
          <lowerUncertainty>0.200000003</lowerUncertainty>
          <upperUncertainty>0.200000003</upperUncertainty>
        </time>
        <waveformID networkCode="4T" stationCode="NM02" locationCode="00" channelCode="HHZ"></waveformID>
        <phaseHint>P</phaseHint>
        <polarity>positive</polarity>
        <evaluationMode>manual</evaluationMode>
        <creationInfo>
          <agencyID>TXNet</agencyID>
          <author>Tricia</author>
          <creationTime>2020-03-26T15:34:52.374539Z</creationTime>
        </creationInfo>
      </pick>
      <pick publicID="smi:org.gfz-potsdam.de/geofon/Pick/20200326153452.374561.258041">
        <time>
          <value>2020-03-26T15:16:41.847597Z</value>
          <lowerUncertainty>0.5</lowerUncertainty>
          <upperUncertainty>0.5</upperUncertainty>
        </time>
        <waveformID networkCode="TX" stationCode="PB06" locationCode="00" channelCode="HHZ"></waveformID>
        <phaseHint>P</phaseHint>
        <evaluationMode>manual</evaluationMode>
        <creationInfo>
          <agencyID>TXNet</agencyID>
          <author>Tricia</author>
          <creationTime>2020-03-26T15:34:52.374587Z</creationTime>
        </creationInfo>
      </pick>
      <pick publicID="smi:org.gfz-potsdam.de/geofon/Pick/20200326153452.374599.258042">
        <time>
          <value>2020-03-26T15:16:52.223552Z</value>
          <lowerUncertainty>0.3000000119</lowerUncertainty>
          <upperUncertainty>0.3000000119</upperUncertainty>
        </time>
        <waveformID networkCode="TX" stationCode="PB06" locationCode="00" channelCode="HHT"></waveformID>
        <phaseHint>S</phaseHint>
        <evaluationMode>manual</evaluationMode>
        <creationInfo>
          <agencyID>TXNet</agencyID>
          <author>Tricia</author>
          <creationTime>2020-03-26T15:34:52.374617Z</creationTime>
        </creationInfo>
      </pick>
      <pick publicID="smi:org.gfz-potsdam.de/geofon/Pick/20200326153452.374629.258043">
        <time>
          <value>2020-03-26T15:16:41.847591Z</value>
          <lowerUncertainty>0.5</lowerUncertainty>
          <upperUncertainty>0.5</upperUncertainty>
        </time>
        <waveformID networkCode="SC" stationCode="CBET" locationCode="00" channelCode="EHZ"></waveformID>
        <phaseHint>P</phaseHint>
        <evaluationMode>manual</evaluationMode>
        <creationInfo>
          <agencyID>TXNet</agencyID>
          <author>Tricia</author>
          <creationTime>2020-03-26T15:34:52.374647Z</creationTime>
        </creationInfo>
      </pick>
      <pick publicID="smi:org.gfz-potsdam.de/geofon/Pick/20200326153452.374666.258044">
        <time>
          <value>2020-03-26T15:16:42.906363Z</value>
          <lowerUncertainty>0.3000000119</lowerUncertainty>
          <upperUncertainty>0.3000000119</upperUncertainty>
        </time>
        <waveformID networkCode="TX" stationCode="PB03" locationCode="00" channelCode="HHZ"></waveformID>
        <phaseHint>P</phaseHint>
        <evaluationMode>manual</evaluationMode>
        <creationInfo>
          <agencyID>TXNet</agencyID>
          <author>Tricia</author>
          <creationTime>2020-03-26T15:34:52.374685Z</creationTime>
        </creationInfo>
      </pick>
      <pick publicID="smi:org.gfz-potsdam.de/geofon/Pick/20200326153452.374697.258045">
        <time>
          <value>2020-03-26T15:16:53.930039Z</value>
          <lowerUncertainty>0.5</lowerUncertainty>
          <upperUncertainty>0.5</upperUncertainty>
        </time>
        <waveformID networkCode="TX" stationCode="PB03" locationCode="00" channelCode="HHT"></waveformID>
        <phaseHint>S</phaseHint>
        <evaluationMode>manual</evaluationMode>
        <creationInfo>
          <agencyID>TXNet</agencyID>
          <author>Tricia</author>
          <creationTime>2020-03-26T15:34:52.374717Z</creationTime>
        </creationInfo>
      </pick>
      <pick publicID="smi:org.gfz-potsdam.de/geofon/Pick/20200326153452.374731.258046">
        <time>
          <value>2020-03-26T15:16:42.831627Z</value>
          <lowerUncertainty>0.3000000119</lowerUncertainty>
          <upperUncertainty>0.3000000119</upperUncertainty>
        </time>
        <waveformID networkCode="TX" stationCode="PB01" locationCode="00" channelCode="HHZ"></waveformID>
        <phaseHint>P</phaseHint>
        <evaluationMode>manual</evaluationMode>
        <creationInfo>
          <agencyID>TXNet</agencyID>
          <author>Tricia</author>
          <creationTime>2020-03-26T15:34:52.374749Z</creationTime>
        </creationInfo>
      </pick>
      <pick publicID="smi:org.gfz-potsdam.de/geofon/Pick/20200326153452.37476.258047">
        <time>
          <value>2020-03-26T15:16:53.992319Z</value>
          <lowerUncertainty>0.400000006</lowerUncertainty>
          <upperUncertainty>0.400000006</upperUncertainty>
        </time>
        <waveformID networkCode="TX" stationCode="PB01" locationCode="00" channelCode="HHT"></waveformID>
        <phaseHint>S</phaseHint>
        <evaluationMode>manual</evaluationMode>
        <creationInfo>
          <agencyID>TXNet</agencyID>
          <author>Tricia</author>
          <creationTime>2020-03-26T15:34:52.374778Z</creationTime>
        </creationInfo>
      </pick>
      <pick publicID="smi:org.gfz-potsdam.de/geofon/Pick/20200326153452.374793.258048">
        <time>
          <value>2020-03-26T15:16:54.328635Z</value>
          <lowerUncertainty>0.400000006</lowerUncertainty>
          <upperUncertainty>0.400000006</upperUncertainty>
        </time>
        <waveformID networkCode="TX" stationCode="PB27" locationCode="00" channelCode="HHT"></waveformID>
        <phaseHint>S</phaseHint>
        <evaluationMode>manual</evaluationMode>
        <creationInfo>
          <agencyID>TXNet</agencyID>
          <author>Tricia</author>
          <creationTime>2020-03-26T15:34:52.374811Z</creationTime>
        </creationInfo>
      </pick>
      <pick publicID="smi:org.gfz-potsdam.de/geofon/Pick/20200326153452.374822.258049">
        <time>
          <value>2020-03-26T15:16:43.242676Z</value>
          <lowerUncertainty>0.400000006</lowerUncertainty>
          <upperUncertainty>0.400000006</upperUncertainty>
        </time>
        <waveformID networkCode="TX" stationCode="PB27" locationCode="00" channelCode="HHZ"></waveformID>
        <phaseHint>P</phaseHint>
        <evaluationMode>manual</evaluationMode>
        <creationInfo>
          <agencyID>TXNet</agencyID>
          <author>Tricia</author>
          <creationTime>2020-03-26T15:34:52.374840Z</creationTime>
        </creationInfo>
      </pick>
      <pick publicID="smi:org.gfz-potsdam.de/geofon/Pick/20200326153452.374853.258050">
        <time>
          <value>2020-03-26T15:16:44.226709Z</value>
          <lowerUncertainty>0.3000000119</lowerUncertainty>
          <upperUncertainty>0.3000000119</upperUncertainty>
        </time>
        <waveformID networkCode="TX" stationCode="PB04" locationCode="00" channelCode="HHZ"></waveformID>
        <phaseHint>P</phaseHint>
        <evaluationMode>manual</evaluationMode>
        <creationInfo>
          <agencyID>TXNet</agencyID>
          <author>Tricia</author>
          <creationTime>2020-03-26T15:34:52.374871Z</creationTime>
        </creationInfo>
      </pick>
      <pick publicID="smi:org.gfz-potsdam.de/geofon/Pick/20200326153452.374882.258051">
        <time>
          <value>2020-03-26T15:16:56.134769Z</value>
          <lowerUncertainty>0.400000006</lowerUncertainty>
          <upperUncertainty>0.400000006</upperUncertainty>
        </time>
        <waveformID networkCode="TX" stationCode="PB04" locationCode="00" channelCode="HHT"></waveformID>
        <phaseHint>S</phaseHint>
        <evaluationMode>manual</evaluationMode>
        <creationInfo>
          <agencyID>TXNet</agencyID>
          <author>Tricia</author>
          <creationTime>2020-03-26T15:34:52.374899Z</creationTime>
        </creationInfo>
      </pick>
      <pick publicID="smi:org.gfz-potsdam.de/geofon/Pick/20200326153452.37491.258052">
        <time>
          <value>2020-03-26T15:16:56.309155Z</value>
          <lowerUncertainty>0.3000000119</lowerUncertainty>
          <upperUncertainty>0.3000000119</upperUncertainty>
        </time>
        <waveformID networkCode="4T" stationCode="NM03" locationCode="00" channelCode="HHT"></waveformID>
        <phaseHint>S</phaseHint>
        <evaluationMode>manual</evaluationMode>
        <creationInfo>
          <agencyID>TXNet</agencyID>
          <author>Tricia</author>
          <creationTime>2020-03-26T15:34:52.374927Z</creationTime>
        </creationInfo>
      </pick>
      <pick publicID="smi:org.gfz-potsdam.de/geofon/Pick/20200326153452.374937.258053">
        <time>
          <value>2020-03-26T15:16:44.189340Z</value>
          <lowerUncertainty>0.400000006</lowerUncertainty>
          <upperUncertainty>0.400000006</upperUncertainty>
        </time>
        <waveformID networkCode="4T" stationCode="NM03" locationCode="00" channelCode="HHZ"></waveformID>
        <phaseHint>P</phaseHint>
        <evaluationMode>manual</evaluationMode>
        <creationInfo>
          <agencyID>TXNet</agencyID>
          <author>Tricia</author>
          <creationTime>2020-03-26T15:34:52.374954Z</creationTime>
        </creationInfo>
      </pick>
      <pick publicID="smi:org.gfz-potsdam.de/geofon/Pick/20200326153452.374966.258054">
        <time>
          <value>2020-03-26T15:16:44.288989Z</value>
          <lowerUncertainty>0.5</lowerUncertainty>
          <upperUncertainty>0.5</upperUncertainty>
        </time>
        <waveformID networkCode="4T" stationCode="NM01" locationCode="00" channelCode="HHZ"></waveformID>
        <phaseHint>P</phaseHint>
        <evaluationMode>manual</evaluationMode>
        <creationInfo>
          <agencyID>TXNet</agencyID>
          <author>Tricia</author>
          <creationTime>2020-03-26T15:34:52.374983Z</creationTime>
        </creationInfo>
      </pick>
      <pick publicID="smi:org.gfz-potsdam.de/geofon/Pick/20200326153452.374992.258055">
        <time>
          <value>2020-03-26T15:16:56.060033Z</value>
          <lowerUncertainty>0.400000006</lowerUncertainty>
          <upperUncertainty>0.400000006</upperUncertainty>
        </time>
        <waveformID networkCode="4T" stationCode="NM01" locationCode="00" channelCode="HHT"></waveformID>
        <phaseHint>S</phaseHint>
        <evaluationMode>manual</evaluationMode>
        <creationInfo>
          <agencyID>TXNet</agencyID>
          <author>Tricia</author>
          <creationTime>2020-03-26T15:34:52.375009Z</creationTime>
        </creationInfo>
      </pick>
      <pick publicID="smi:org.gfz-potsdam.de/geofon/Pick/20200326153452.375021.258056">
        <time>
          <value>2020-03-26T15:16:44.637768Z</value>
          <lowerUncertainty>0.5</lowerUncertainty>
          <upperUncertainty>0.5</upperUncertainty>
        </time>
        <waveformID networkCode="SC" stationCode="SRH" locationCode="00" channelCode="EHZ"></waveformID>
        <phaseHint>P</phaseHint>
        <evaluationMode>manual</evaluationMode>
        <creationInfo>
          <agencyID>TXNet</agencyID>
          <author>Tricia</author>
          <creationTime>2020-03-26T15:34:52.375039Z</creationTime>
        </creationInfo>
      </pick>
      <pick publicID="smi:org.gfz-potsdam.de/geofon/Pick/20200326153452.375053.258057">
        <time>
          <value>2020-03-26T15:16:45.111100Z</value>
          <lowerUncertainty>0.3000000119</lowerUncertainty>
          <upperUncertainty>0.3000000119</upperUncertainty>
        </time>
        <waveformID networkCode="TX" stationCode="PB16" locationCode="00" channelCode="HHZ"></waveformID>
        <phaseHint>P</phaseHint>
        <evaluationMode>manual</evaluationMode>
        <creationInfo>
          <agencyID>TXNet</agencyID>
          <author>Tricia</author>
          <creationTime>2020-03-26T15:34:52.375071Z</creationTime>
        </creationInfo>
      </pick>
      <pick publicID="smi:org.gfz-potsdam.de/geofon/Pick/20200326153452.375082.258058">
        <time>
          <value>2020-03-26T15:16:57.430213Z</value>
          <lowerUncertainty>0.3000000119</lowerUncertainty>
          <upperUncertainty>0.3000000119</upperUncertainty>
        </time>
        <waveformID networkCode="TX" stationCode="PB16" locationCode="00" channelCode="HHT"></waveformID>
        <phaseHint>S</phaseHint>
        <evaluationMode>manual</evaluationMode>
        <creationInfo>
          <agencyID>TXNet</agencyID>
          <author>Tricia</author>
          <creationTime>2020-03-26T15:34:52.375099Z</creationTime>
        </creationInfo>
      </pick>
      <pick publicID="smi:org.gfz-potsdam.de/geofon/Pick/20200326153452.375112.258059">
        <time>
          <value>2020-03-26T15:16:45.023901Z</value>
          <lowerUncertainty>0.5</lowerUncertainty>
          <upperUncertainty>0.5</upperUncertainty>
        </time>
        <waveformID networkCode="TX" stationCode="PB19" locationCode="00" channelCode="HHZ"></waveformID>
        <phaseHint>P</phaseHint>
        <evaluationMode>manual</evaluationMode>
        <creationInfo>
          <agencyID>TXNet</agencyID>
          <author>Tricia</author>
          <creationTime>2020-03-26T15:34:52.375130Z</creationTime>
        </creationInfo>
      </pick>
      <pick publicID="smi:org.gfz-potsdam.de/geofon/Pick/20200326153452.375144.258060">
        <time>
          <value>2020-03-26T15:16:45.335305Z</value>
          <lowerUncertainty>0.3000000119</lowerUncertainty>
          <upperUncertainty>0.3000000119</upperUncertainty>
        </time>
        <waveformID networkCode="TX" stationCode="PB21" locationCode="00" channelCode="HHZ"></waveformID>
        <phaseHint>P</phaseHint>
        <evaluationMode>manual</evaluationMode>
        <creationInfo>
          <agencyID>TXNet</agencyID>
          <author>Tricia</author>
          <creationTime>2020-03-26T15:34:52.375161Z</creationTime>
        </creationInfo>
      </pick>
      <pick publicID="smi:org.gfz-potsdam.de/geofon/Pick/20200326153452.375175.258061">
        <time>
          <value>2020-03-26T15:16:57.442664Z</value>
          <lowerUncertainty>0.400000006</lowerUncertainty>
          <upperUncertainty>0.400000006</upperUncertainty>
        </time>
        <waveformID networkCode="TX" stationCode="PB21" locationCode="00" channelCode="HHT"></waveformID>
        <phaseHint>S</phaseHint>
        <evaluationMode>manual</evaluationMode>
        <creationInfo>
          <agencyID>TXNet</agencyID>
          <author>Tricia</author>
          <creationTime>2020-03-26T15:34:52.375194Z</creationTime>
        </creationInfo>
      </pick>
      <pick publicID="smi:org.gfz-potsdam.de/geofon/Pick/20200326153452.375208.258062">
        <time>
          <value>2020-03-26T15:16:46.070219Z</value>
          <lowerUncertainty>0.400000006</lowerUncertainty>
          <upperUncertainty>0.400000006</upperUncertainty>
        </time>
        <waveformID networkCode="TX" stationCode="PB15" locationCode="00" channelCode="HHZ"></waveformID>
        <phaseHint>P</phaseHint>
        <evaluationMode>manual</evaluationMode>
        <creationInfo>
          <agencyID>TXNet</agencyID>
          <author>Tricia</author>
          <creationTime>2020-03-26T15:34:52.375226Z</creationTime>
        </creationInfo>
      </pick>
      <pick publicID="smi:org.gfz-potsdam.de/geofon/Pick/20200326153452.37524.258063">
        <time>
          <value>2020-03-26T15:17:00.668804Z</value>
          <lowerUncertainty>0.3000000119</lowerUncertainty>
          <upperUncertainty>0.3000000119</upperUncertainty>
        </time>
        <waveformID networkCode="TX" stationCode="PB05" locationCode="00" channelCode="HHT"></waveformID>
        <phaseHint>S</phaseHint>
        <evaluationMode>manual</evaluationMode>
        <creationInfo>
          <agencyID>TXNet</agencyID>
          <author>Tricia</author>
          <creationTime>2020-03-26T15:34:52.375320Z</creationTime>
        </creationInfo>
      </pick>
      <pick publicID="smi:org.gfz-potsdam.de/geofon/Pick/20200326153452.375353.258064">
        <time>
          <value>2020-03-26T15:16:46.717940Z</value>
          <lowerUncertainty>0.5</lowerUncertainty>
          <upperUncertainty>0.5</upperUncertainty>
        </time>
        <waveformID networkCode="TX" stationCode="PB05" locationCode="00" channelCode="HHZ"></waveformID>
        <phaseHint>P</phaseHint>
        <evaluationMode>manual</evaluationMode>
        <creationInfo>
          <agencyID>TXNet</agencyID>
          <author>Tricia</author>
          <creationTime>2020-03-26T15:34:52.375379Z</creationTime>
        </creationInfo>
      </pick>
      <pick publicID="smi:org.gfz-potsdam.de/geofon/Pick/20200326153452.375392.258065">
        <time>
          <value>2020-03-26T15:16:47.452853Z</value>
          <lowerUncertainty>0.5</lowerUncertainty>
          <upperUncertainty>0.5</upperUncertainty>
        </time>
        <waveformID networkCode="SC" stationCode="DAG" locationCode="00" channelCode="EHZ"></waveformID>
        <phaseHint>P</phaseHint>
        <evaluationMode>manual</evaluationMode>
        <creationInfo>
          <agencyID>TXNet</agencyID>
          <author>Tricia</author>
          <creationTime>2020-03-26T15:34:52.375411Z</creationTime>
        </creationInfo>
      </pick>
      <pick publicID="smi:org.gfz-potsdam.de/geofon/Pick/20200326153452.375438.258066">
        <time>
          <value>2020-03-26T15:17:04.617401Z</value>
          <lowerUncertainty>0.5</lowerUncertainty>
          <upperUncertainty>0.5</upperUncertainty>
        </time>
        <waveformID networkCode="US" stationCode="MNTX" locationCode="00" channelCode="BHT"></waveformID>
        <phaseHint>S</phaseHint>
        <evaluationMode>manual</evaluationMode>
        <creationInfo>
          <agencyID>TXNet</agencyID>
          <author>Tricia</author>
          <creationTime>2020-03-26T15:34:52.375457Z</creationTime>
        </creationInfo>
      </pick>
      <pick publicID="smi:org.gfz-potsdam.de/geofon/Pick/20200326153452.375476.258067">
        <time>
          <value>2020-03-26T15:17:07.929961Z</value>
          <lowerUncertainty>0.5</lowerUncertainty>
          <upperUncertainty>0.5</upperUncertainty>
        </time>
        <waveformID networkCode="TX" stationCode="VHRN" locationCode="00" channelCode="HHT"></waveformID>
        <phaseHint>S</phaseHint>
        <evaluationMode>manual</evaluationMode>
        <creationInfo>
          <agencyID>TXNet</agencyID>
          <author>Tricia</author>
          <creationTime>2020-03-26T15:34:52.375493Z</creationTime>
        </creationInfo>
      </pick>
      <pick publicID="smi:org.gfz-potsdam.de/geofon/Pick/20200326153452.375506.258068">
        <time>
          <value>2020-03-26T15:17:08.609161Z</value>
          <lowerUncertainty>0.400000006</lowerUncertainty>
          <upperUncertainty>0.400000006</upperUncertainty>
        </time>
        <waveformID networkCode="TX" stationCode="PB08" locationCode="00" channelCode="HHT"></waveformID>
        <phaseHint>S</phaseHint>
        <evaluationMode>manual</evaluationMode>
        <creationInfo>
          <agencyID>TXNet</agencyID>
          <author>Tricia</author>
          <creationTime>2020-03-26T15:34:52.375524Z</creationTime>
        </creationInfo>
      </pick>
      <pick publicID="smi:org.gfz-potsdam.de/geofon/Pick/20200326153452.375539.258069">
        <time>
          <value>2020-03-26T15:16:51.464521Z</value>
          <lowerUncertainty>0.400000006</lowerUncertainty>
          <upperUncertainty>0.400000006</upperUncertainty>
        </time>
        <waveformID networkCode="TX" stationCode="PB08" locationCode="00" channelCode="HHZ"></waveformID>
        <phaseHint>P</phaseHint>
        <evaluationMode>manual</evaluationMode>
        <creationInfo>
          <agencyID>TXNet</agencyID>
          <author>Tricia</author>
          <creationTime>2020-03-26T15:34:52.375557Z</creationTime>
        </creationInfo>
      </pick>
      <pick publicID="smi:org.gfz-potsdam.de/geofon/Pick/20200326153452.375571.258070">
        <time>
          <value>2020-03-26T15:16:52.740584Z</value>
          <lowerUncertainty>0.5</lowerUncertainty>
          <upperUncertainty>0.5</upperUncertainty>
        </time>
        <waveformID networkCode="SC" stationCode="CPRX" locationCode="00" channelCode="EHZ"></waveformID>
        <phaseHint>P</phaseHint>
        <evaluationMode>manual</evaluationMode>
        <creationInfo>
          <agencyID>TXNet</agencyID>
          <author>Tricia</author>
          <creationTime>2020-03-26T15:34:52.375589Z</creationTime>
        </creationInfo>
      </pick>
      <pick publicID="smi:org.gfz-potsdam.de/geofon/Pick/20200326153452.375605.258071">
        <time>
          <value>2020-03-26T15:16:52.678839Z</value>
          <lowerUncertainty>0.400000006</lowerUncertainty>
          <upperUncertainty>0.400000006</upperUncertainty>
        </time>
        <waveformID networkCode="TX" stationCode="ODSA" locationCode="00" channelCode="HHZ"></waveformID>
        <phaseHint>P</phaseHint>
        <evaluationMode>manual</evaluationMode>
        <creationInfo>
          <agencyID>TXNet</agencyID>
          <author>Tricia</author>
          <creationTime>2020-03-26T15:34:52.375623Z</creationTime>
        </creationInfo>
      </pick>
      <pick publicID="smi:org.gfz-potsdam.de/geofon/Pick/20200326153452.375635.258072">
        <time>
          <value>2020-03-26T15:17:10.564431Z</value>
          <lowerUncertainty>0.400000006</lowerUncertainty>
          <upperUncertainty>0.400000006</upperUncertainty>
        </time>
        <waveformID networkCode="TX" stationCode="ODSA" locationCode="00" channelCode="HHT"></waveformID>
        <phaseHint>S</phaseHint>
        <evaluationMode>manual</evaluationMode>
        <creationInfo>
          <agencyID>TXNet</agencyID>
          <author>Tricia</author>
          <creationTime>2020-03-26T15:34:52.375654Z</creationTime>
        </creationInfo>
      </pick>
      <pick publicID="smi:org.gfz-potsdam.de/geofon/Pick/20200326153452.375671.258073">
        <time>
          <value>2020-03-26T15:17:11.305376Z</value>
          <lowerUncertainty>0.400000006</lowerUncertainty>
          <upperUncertainty>0.400000006</upperUncertainty>
        </time>
        <waveformID networkCode="TX" stationCode="ALPN" locationCode="00" channelCode="HHR"></waveformID>
        <phaseHint>S</phaseHint>
        <evaluationMode>manual</evaluationMode>
        <creationInfo>
          <agencyID>TXNet</agencyID>
          <author>Tricia</author>
          <creationTime>2020-03-26T15:34:52.375688Z</creationTime>
        </creationInfo>
      </pick>
      <pick publicID="smi:org.gfz-potsdam.de/geofon/Pick/20200326153452.375699.258074">
        <time>
          <value>2020-03-26T15:16:52.802329Z</value>
          <lowerUncertainty>0.400000006</lowerUncertainty>
          <upperUncertainty>0.400000006</upperUncertainty>
        </time>
        <waveformID networkCode="TX" stationCode="ALPN" locationCode="00" channelCode="HHZ"></waveformID>
        <phaseHint>P</phaseHint>
        <evaluationMode>manual</evaluationMode>
        <creationInfo>
          <agencyID>TXNet</agencyID>
          <author>Tricia</author>
          <creationTime>2020-03-26T15:34:52.375717Z</creationTime>
        </creationInfo>
      </pick>
      <pick publicID="smi:org.gfz-potsdam.de/geofon/Pick/20200326153452.375731.258075">
        <time>
          <value>2020-03-26T15:16:56.589382Z</value>
          <lowerUncertainty>0.400000006</lowerUncertainty>
          <upperUncertainty>0.400000006</upperUncertainty>
        </time>
        <waveformID networkCode="TX" stationCode="MB07" locationCode="00" channelCode="HHZ"></waveformID>
        <phaseHint>P</phaseHint>
        <evaluationMode>manual</evaluationMode>
        <creationInfo>
          <agencyID>TXNet</agencyID>
          <author>Tricia</author>
          <creationTime>2020-03-26T15:34:52.375748Z</creationTime>
        </creationInfo>
      </pick>
      <pick publicID="smi:org.gfz-potsdam.de/geofon/Pick/20200326153452.375766.258076">
        <time>
          <value>2020-03-26T15:17:23.963189Z</value>
          <lowerUncertainty>0.400000006</lowerUncertainty>
          <upperUncertainty>0.400000006</upperUncertainty>
        </time>
        <waveformID networkCode="TX" stationCode="MB02" locationCode="00" channelCode="HHT"></waveformID>
        <phaseHint>S</phaseHint>
        <evaluationMode>manual</evaluationMode>
        <creationInfo>
          <agencyID>TXNet</agencyID>
          <author>Tricia</author>
          <creationTime>2020-03-26T15:34:52.375784Z</creationTime>
        </creationInfo>
      </pick>
      <pick publicID="smi:org.gfz-potsdam.de/geofon/Pick/20200326153452.375795.258077">
        <time>
          <value>2020-03-26T15:17:00.417591Z</value>
          <lowerUncertainty>0.400000006</lowerUncertainty>
          <upperUncertainty>0.400000006</upperUncertainty>
        </time>
        <waveformID networkCode="TX" stationCode="MB02" locationCode="00" channelCode="HHZ"></waveformID>
        <phaseHint>P</phaseHint>
        <evaluationMode>manual</evaluationMode>
        <creationInfo>
          <agencyID>TXNet</agencyID>
          <author>Tricia</author>
          <creationTime>2020-03-26T15:34:52.375812Z</creationTime>
        </creationInfo>
      </pick>
      <pick publicID="smi:org.gfz-potsdam.de/geofon/Pick/20200326153452.375825.258078">
        <time>
          <value>2020-03-26T15:17:02.866826Z</value>
          <lowerUncertainty>0.400000006</lowerUncertainty>
          <upperUncertainty>0.400000006</upperUncertainty>
        </time>
        <waveformID networkCode="TX" stationCode="MB06" locationCode="00" channelCode="HHZ"></waveformID>
        <phaseHint>P</phaseHint>
        <evaluationMode>manual</evaluationMode>
        <creationInfo>
          <agencyID>TXNet</agencyID>
          <author>Tricia</author>
          <creationTime>2020-03-26T15:34:52.375843Z</creationTime>
        </creationInfo>
      </pick>
      <pick publicID="smi:org.gfz-potsdam.de/geofon/Pick/20200326153452.375855.258079">
        <time>
          <value>2020-03-26T15:17:27.811982Z</value>
          <lowerUncertainty>0.5</lowerUncertainty>
          <upperUncertainty>0.5</upperUncertainty>
        </time>
        <waveformID networkCode="TX" stationCode="MB06" locationCode="00" channelCode="HHT"></waveformID>
        <phaseHint>S</phaseHint>
        <evaluationMode>manual</evaluationMode>
        <creationInfo>
          <agencyID>TXNet</agencyID>
          <author>Tricia</author>
          <creationTime>2020-03-26T15:34:52.375872Z</creationTime>
        </creationInfo>
      </pick>
      <pick publicID="smi:org.gfz-potsdam.de/geofon/Pick/20200326153452.375887.258080">
        <time>
          <value>2020-03-26T15:17:34.418743Z</value>
          <lowerUncertainty>0.5</lowerUncertainty>
          <upperUncertainty>0.5</upperUncertainty>
        </time>
        <waveformID networkCode="EP" stationCode="KIDD" channelCode="HHT"></waveformID>
        <phaseHint>S</phaseHint>
        <evaluationMode>manual</evaluationMode>
        <creationInfo>
          <agencyID>TXNet</agencyID>
          <author>Tricia</author>
          <creationTime>2020-03-26T15:34:52.375904Z</creationTime>
        </creationInfo>
      </pick>
      <pick publicID="smi:org.gfz-potsdam.de/geofon/Pick/20200326153452.375945.258082">
        <time>
          <value>2020-03-26T15:17:32.689871Z</value>
          <lowerUncertainty>0.6000000238</lowerUncertainty>
          <upperUncertainty>0.6000000238</upperUncertainty>
        </time>
        <waveformID networkCode="TX" stationCode="MB05" locationCode="00" channelCode="HHT"></waveformID>
        <phaseHint>S</phaseHint>
        <evaluationMode>manual</evaluationMode>
        <creationInfo>
          <agencyID>TXNet</agencyID>
          <author>Tricia</author>
          <creationTime>2020-03-26T15:34:52.375962Z</creationTime>
        </creationInfo>
      </pick>
      <pick publicID="smi:org.gfz-potsdam.de/geofon/Pick/20200326153452.375973.258083">
        <time>
          <value>2020-03-26T15:17:04.698606Z</value>
          <lowerUncertainty>0.6000000238</lowerUncertainty>
          <upperUncertainty>0.6000000238</upperUncertainty>
        </time>
        <waveformID networkCode="TX" stationCode="MB05" locationCode="00" channelCode="HHZ"></waveformID>
        <phaseHint>P</phaseHint>
        <evaluationMode>manual</evaluationMode>
        <creationInfo>
          <agencyID>TXNet</agencyID>
          <author>Tricia</author>
          <creationTime>2020-03-26T15:34:52.375991Z</creationTime>
        </creationInfo>
      </pick>
      <pick publicID="smi:org.gfz-potsdam.de/geofon/Pick/20200326153452.37601.258084">
        <time>
          <value>2020-03-26T15:17:08.979622Z</value>
          <lowerUncertainty>0.6999999881</lowerUncertainty>
          <upperUncertainty>0.6999999881</upperUncertainty>
        </time>
        <waveformID networkCode="IM" stationCode="TX31" channelCode="BHZ"></waveformID>
        <phaseHint>P</phaseHint>
        <evaluationMode>manual</evaluationMode>
        <creationInfo>
          <agencyID>TXNet</agencyID>
          <author>Tricia</author>
          <creationTime>2020-03-26T15:34:52.376029Z</creationTime>
        </creationInfo>
      </pick>
      <pick publicID="smi:org.gfz-potsdam.de/geofon/Pick/20200326154509.418524.259129">
        <time>
          <value>2020-03-26T15:16:36.862431Z</value>
          <lowerUncertainty>0.200000003</lowerUncertainty>
          <upperUncertainty>0.200000003</upperUncertainty>
        </time>
        <waveformID networkCode="TX" stationCode="PB13" locationCode="00" channelCode="HHT"></waveformID>
        <phaseHint>S</phaseHint>
        <evaluationMode>manual</evaluationMode>
        <creationInfo>
          <agencyID>TXNet</agencyID>
          <author>Tricia</author>
          <creationTime>2020-03-26T15:45:09.418605Z</creationTime>
        </creationInfo>
      </pick>
      <pick publicID="smi:org.gfz-potsdam.de/geofon/Pick/20200326154509.418628.259130">
        <time>
          <value>2020-03-26T15:16:40.649484Z</value>
          <lowerUncertainty>0.400000006</lowerUncertainty>
          <upperUncertainty>0.400000006</upperUncertainty>
        </time>
        <waveformID networkCode="TX" stationCode="PB07" locationCode="00" channelCode="HHT"></waveformID>
        <phaseHint>S</phaseHint>
        <evaluationMode>manual</evaluationMode>
        <creationInfo>
          <agencyID>TXNet</agencyID>
          <author>Tricia</author>
          <creationTime>2020-03-26T15:45:09.418650Z</creationTime>
        </creationInfo>
      </pick>
      <pick publicID="smi:org.gfz-potsdam.de/geofon/Pick/20200326154509.41868.259131">
        <time>
          <value>2020-03-26T15:16:40.752393Z</value>
          <lowerUncertainty>0.400000006</lowerUncertainty>
          <upperUncertainty>0.400000006</upperUncertainty>
        </time>
        <waveformID networkCode="TX" stationCode="PECS" locationCode="00" channelCode="HHT"></waveformID>
        <phaseHint>S</phaseHint>
        <evaluationMode>manual</evaluationMode>
        <creationInfo>
          <agencyID>TXNet</agencyID>
          <author>Tricia</author>
          <creationTime>2020-03-26T15:45:09.418718Z</creationTime>
        </creationInfo>
      </pick>
      <pick publicID="smi:org.gfz-potsdam.de/geofon/Pick/20200326154509.418812.259132">
        <time>
          <value>2020-03-26T15:17:05.203584Z</value>
          <lowerUncertainty>0.400000006</lowerUncertainty>
          <upperUncertainty>0.400000006</upperUncertainty>
        </time>
        <waveformID networkCode="TX" stationCode="MNHN" locationCode="00" channelCode="HHT"></waveformID>
        <phaseHint>S</phaseHint>
        <evaluationMode>manual</evaluationMode>
        <creationInfo>
          <agencyID>TXNet</agencyID>
          <author>Tricia</author>
          <creationTime>2020-03-26T15:45:09.418836Z</creationTime>
        </creationInfo>
      </pick>
      <pick publicID="smi:org.gfz-potsdam.de/geofon/Pick/20200326154509.418863.259133">
        <time>
          <value>2020-03-26T15:17:16.708815Z</value>
          <lowerUncertainty>0.5</lowerUncertainty>
          <upperUncertainty>0.5</upperUncertainty>
        </time>
        <waveformID networkCode="TX" stationCode="MB07" locationCode="00" channelCode="HHT"></waveformID>
        <phaseHint>S</phaseHint>
        <evaluationMode>manual</evaluationMode>
        <creationInfo>
          <agencyID>TXNet</agencyID>
          <author>Tricia</author>
          <creationTime>2020-03-26T15:45:09.418883Z</creationTime>
        </creationInfo>
      </pick>
      <pick publicID="smi:org.gfz-potsdam.de/geofon/Pick/20200326154509.418911.259134">
        <time>
          <value>2020-03-26T15:17:05.080087Z</value>
          <lowerUncertainty>0.5</lowerUncertainty>
          <upperUncertainty>0.5</upperUncertainty>
        </time>
        <waveformID networkCode="EP" stationCode="KIDD" channelCode="HHZ"></waveformID>
        <phaseHint>P</phaseHint>
        <evaluationMode>manual</evaluationMode>
        <creationInfo>
          <agencyID>TXNet</agencyID>
          <author>Tricia</author>
          <creationTime>2020-03-26T15:45:09.418929Z</creationTime>
        </creationInfo>
      </pick>
      <pick publicID="smi:org.gfz-potsdam.de/geofon/Pick/20200416191248.529211.62721">
        <time>
          <value>2020-03-26T15:16:36.759126Z</value>
          <lowerUncertainty>0.200000003</lowerUncertainty>
          <upperUncertainty>0.200000003</upperUncertainty>
        </time>
        <waveformID networkCode="TX" stationCode="PB09" locationCode="00" channelCode="HHT"></waveformID>
        <phaseHint>S</phaseHint>
        <evaluationMode>manual</evaluationMode>
        <creationInfo>
          <agencyID>TXNet</agencyID>
          <author>Tricia</author>
          <creationTime>2020-04-16T19:12:48.529327Z</creationTime>
        </creationInfo>
      </pick>
      <pick publicID="smi:org.gfz-potsdam.de/geofon/Pick/20200416191248.529382.62722">
        <time>
          <value>2020-03-26T15:16:33.296820Z</value>
          <lowerUncertainty>0.200000003</lowerUncertainty>
          <upperUncertainty>0.200000003</upperUncertainty>
        </time>
        <waveformID networkCode="TX" stationCode="PB09" locationCode="00" channelCode="HHZ"></waveformID>
        <phaseHint>P</phaseHint>
        <polarity>positive</polarity>
        <evaluationMode>manual</evaluationMode>
        <creationInfo>
          <agencyID>TXNet</agencyID>
          <author>Tricia</author>
          <creationTime>2020-04-16T19:12:48.529406Z</creationTime>
        </creationInfo>
      </pick>
    </event>
  </eventParameters>
</q:quakeml>
