# Compuver Vision 이상치 탐지 알고리즘 경진대회

# 목표
불균형 데이터 셋을 학습하여 사물의 상태를 잘 분류할 수 있는 알고리즘 만들기


# EDA
정상 샘플과 이상치 샘플이 동시에 존재하므로 Supervised Anomaly Detection 방법으로 접근도 가능하다.  

### Class-State                         |                          Count
--------------------------------------------------------------------------------
001. hazelnut-good                  |                            391
002. screw-good                     |                            320
003. carpet-good                    |                            280
004. pill-good                      |                            267
005. grid-good                      |                            264
006. wood-good                      |                            247
007. leather-good                   |                            245
008. zipper-good                    |                            240
009. tile-good                      |                            230
010. cable-good                     |                            224
011. metal_nut-good                 |                            220
012. capsule-good                   |                            219
013. transistor-good                |                            213
014. bottle-good                    |                            209
015. toothbrush-good                |                             60
016. toothbrush-defective           |                             15
017. metal_nut-bent                 |                             13
018. pill-color                     |                             13
019. pill-crack                     |                             13
020. screw-scratch_neck             |                             13
021. capsule-crack                  |                             12
022. capsule-scratch                |                             12
023. metal_nut-flip                 |                             12
024. metal_nut-scratch              |                             12
025. pill-scratch                   |                             12
026. screw-manipulated_front        |                             12
027. screw-scratch_head             |                             12
028. screw-thread_side              |                             12
029. screw-thread_top               |                             12
030. bottle-broken_small            |                             11
031. bottle-contamination           |                             11
032. capsule-faulty_imprint         |                             11
033. capsule-poke                   |                             11
034. metal_nut-color                |                             11
035. pill-contamination             |                             11
036. wood-scratch                   |                             11
037. bottle-broken_large            |                             10
038. capsule-squeeze                |                             10
039. carpet-color                   |                             10
040. carpet-thread                  |                             10
041. leather-color                  |                             10
042. leather-cut                    |                             10
043. leather-glue                   |                             10
044. pill-faulty_imprint            |                             10
045. zipper-broken_teeth            |                             10
046. carpet-cut                     |                              9
047. carpet-hole                    |                              9
048. carpet-metal_contamination     |                              9
049. hazelnut-crack                 |                              9
050. hazelnut-cut                   |                              9
051. hazelnut-hole                  |                              9
052. hazelnut-print                 |                              9
053. leather-fold                   |                              9
054. leather-poke                   |                              9
055. pill-combined                  |                              9
056. tile-crack                     |                              9
057. tile-glue_strip                |                              9
058. tile-oil                       |                              9
059. zipper-fabric_border           |                              9
060. zipper-rough                   |                              9
061. zipper-split_teeth             |                              9
062. tile-gray_stroke               |                              8
063. tile-rough                     |                              8
064. zipper-combined                |                              8
065. zipper-fabric_interior         |                              8
066. zipper-squeezed_teeth          |                              8
067. cable-bent_wire                |                              7
068. cable-cut_inner_insulation     |                              7
069. cable-cable_swap               |                              6
070. cable-combined                 |                              6
071. cable-missing_cable            |                              6
072. grid-bent                      |                              6
073. grid-broken                    |                              6
074. grid-glue                      |                              6
075. grid-metal_contamination       |                              6
076. grid-thread                    |                              6
077. wood-combined                  |                              6
078. cable-cut_outer_insulation     |                              5
079. cable-missing_wire             |                              5
080. cable-poke_insulation          |                              5
081. pill-pill_type                 |                              5
082. transistor-bent_lead           |                              5
083. transistor-cut_lead            |                              5
084. transistor-damaged_case        |                              5
085. transistor-misplaced           |                              5
086. wood-hole                      |                              5
087. wood-liquid                    |                              5
088. wood-color                     |                              4
