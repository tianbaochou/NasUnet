from util.genotype import Genotype

'''
Search on Pascal voc
'''


# enhance
NAS_UNET_V1_En = Genotype(down=[('down_conv', 0), ('down_dil_conv', 1), ('down_dep_conv', 0), ('down_dep_conv', 1)], down_concat=range(2, 4),
                          up=[('cweight', 0), ('up_cweight', 1), ('conv', 0), ('up_cweight', 1)], up_concat=range(2, 4))
# enhance + sharing
NAS_UNET_V1_En_sh = Genotype(down=[('down_cweight', 0), ('down_cweight', 1), ('down_dep_conv', 0), ('down_cweight', 1)], down_concat=range(2, 4),
         up=[('dep_conv', 0), ('up_cweight', 1), ('cweight', 0), ('up_cweight', 1)], up_concat=range(2, 4))

# enhance
NAS_UNET_V2_En = Genotype(down=[('down_dep_conv', 0), ('down_dil_conv', 1), ('down_cweight', 0), ('down_dep_conv', 1), ('down_dep_conv', 1), ('down_dep_conv', 0)], down_concat=range(2, 5),
                          up=[('identity', 0), ('up_dep_conv', 1), ('cweight', 0), ('up_cweight', 1), ('conv', 2), ('up_cweight', 1)], up_concat=range(2, 5))

NAS_UNET_V2 = Genotype(down=[('down_conv', 1), ('down_dep_conv', 0), ('down_cweight', 1), ('down_dil_conv', 0), ('down_dil_conv', 1), ('down_conv', 0)], down_concat=range(2, 5),
         up=[('identity', 0), ('up_cweight', 1), ('identity', 2), ('up_cweight', 1), ('cweight', 3), ('up_conv', 1)], up_concat=range(2, 5))


NAS_UNET_V3 = Genotype(
    down=[('down_dil_conv', 1), ('down_cweight', 0), ('down_cweight', 0), ('down_cweight', 1), ('down_cweight', 0),
          ('conv', 3), ('down_cweight', 0), ('conv', 4)], down_concat=range(2, 6),
    up=[('cweight', 0), ('up_cweight', 1), ('conv', 2), ('up_cweight', 1), ('up_cweight', 1), ('conv', 3),
        ('up_cweight', 1), ('conv', 4)], up_concat=range(2, 6))

# enhance + no sharing
NAS_UNET_V3_En_sh = Genotype(down=[('down_dep_conv', 0), ('down_cweight', 1), ('conv', 2), ('down_cweight', 1), ('identity', 3), ('down_cweight', 1), ('down_dil_conv', 1), ('conv', 3)], down_concat=range(2, 6),
                              up=[('cweight', 0), ('up_conv', 1), ('cweight', 2), ('up_conv', 1), ('cweight', 3), ('up_conv', 1), ('cweight', 0), ('up_cweight', 1)], up_concat=range(2, 6))

NAS_UNET_NEW_V3 = Genotype(down=[('down_dep_conv', 0), ('down_cweight', 1), ('down_conv', 1), ('max_pool', 0), ('max_pool', 1), ('cweight', 2), ('down_dil_conv', 0), ('down_dil_conv', 1)], down_concat=range(2, 6),
                           up=[('dep_conv', 0), ('up_conv', 1), ('shuffle_conv', 0), ('up_cweight', 1), ('identity', 2), ('up_cweight', 1), ('dil_conv', 3), ('up_cweight', 1)], up_concat=range(2, 6))


NAS_UNET_NEW_V2 =  Genotype(down=[('down_dil_conv', 1), ('down_dep_conv', 0), ('max_pool', 0), ('down_conv', 1), ('down_conv', 1), ('down_dil_conv', 0)], down_concat=range(2, 5),
                            up=[('identity', 0), ('up_dil_conv', 1), ('identity', 0), ('up_dil_conv', 1), ('dil_conv', 3), ('up_cweight', 1)], up_concat=range(2, 5))

NAS_UNET_NEW_V1 = Genotype(down=[('down_dil_conv', 0), ('down_conv', 1), ('max_pool', 1), ('down_conv', 0)], down_concat=range(2, 4),
                           up=[('conv', 0), ('up_dil_conv', 1), ('conv', 2), ('up_cweight', 1)], up_concat=range(2, 4))


NASUNET = NAS_UNET_V2