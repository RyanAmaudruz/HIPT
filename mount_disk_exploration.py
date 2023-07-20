import os
import pandas as pd
import h5py

# def prepare_dir(file_path):
#     """
#     This function is used to create the directories needed to output a path. If the directories already exist, the
#     function continues.
#     """
#     # Remove the file name to only keep the directory path.
#     dir_path = '/'.join(file_path.split('/')[:-1])
#     # Try to create the directory. Will have no effect if the directory already exists.
#     try:
#         os.makedirs(dir_path)
#     except FileExistsError:
#         pass


data_path = '/mnt/disks/data_dir/data/gdc/'

dir_list = os.listdir(data_path)


data_dic = {}
for d in dir_list:
    data_dic[d] = os.listdir(data_path + d)


all_files = []
for file_list in data_dic.values():
    all_files += file_list

all_files2 = [f for f in all_files if f!= 'logs']

extensions = list(set([f.split('.')[-1] for f in all_files2]))


data_list = []
for k, v in data_dic.items():
    for f in v:
        if f != 'logs':

            data_list.append({
                'folder': k,
                'file_name': f,
                'extension': f.split('.')[-1]
            })

df = pd.DataFrame(data_list)


df_svs = df[df['extension'] == 'svs'].reset_index(drop=True)

df_svs['id'] = df_svs['file_name'].map(lambda x: x.split('.')[0])
df_svs['uuid'] = df_svs['file_name'].map(lambda x: x.split('.')[1])
df_svs['project'] = df_svs['id'].map(lambda x: x.split('-')[0])
df_svs['tss'] = df_svs['id'].map(lambda x: x.split('-')[1])
df_svs['participant'] = df_svs['id'].map(lambda x: x.split('-')[2])
df_svs['sample_and_vial'] = df_svs['id'].map(lambda x: x.split('-')[3])
df_svs['portion'] = df_svs['id'].map(lambda x: x.split('-')[4])
df_svs['slide'] = df_svs['id'].map(lambda x: x.split('-')[5])


f_path = 'tcga_code_tables/diseaseStudy.tsv'
with open(f_path, 'r') as f:
    lines = f.readlines()

data_list = []
for l in lines[1:]:
    l_split = l.split('\t')
    data_list.append({
        'study_abbr': l_split[0],
        'study_name': l_split[1].strip(),
    })
disease_study_df = pd.DataFrame(data_list)

study_name_to_abbr_map = dict(zip(disease_study_df['study_name'], disease_study_df['study_abbr']))



f_path = 'tcga_code_tables/tissueSourceSite.tsv'
with open(f_path, 'r') as f:
    lines = f.readlines()

data_list = []
for l in lines[1:]:
    l_split = l.split('\t')
    data_list.append({
        'tss_code': l_split[0],
        'source_site': l_split[1],
        'study_name': l_split[2],
        'brc': l_split[3].strip(),
    })
tissue_source_site_df = pd.DataFrame(data_list)

study_map = dict(zip(tissue_source_site_df['tss_code'], tissue_source_site_df['study_name']))

df_svs['study_name'] = df_svs['tss'].map(study_map)

df_svs['study_abbr'] = df_svs['study_name'].map(study_name_to_abbr_map)


exi_tcga_path = '3-Self-Supervised-Eval/embeddings_slide_lib/embeddings_slide_lib/vit256mean_tcga_slide_embeddings/'
tcga_slide_emb_list = os.listdir(exi_tcga_path)
tcga_slide_emb_df = pd.DataFrame({'file_name': tcga_slide_emb_list})
tcga_slide_emb_df['id'] = tcga_slide_emb_df['file_name'].map(lambda x: x.split('.')[0])

my_tcga_path = '/mnt/disks/data_dir/data/features_4096_fp/pt_files/'
my_tcga_slide_emb_list = os.listdir(my_tcga_path)
tcga_slide_emb_df2 = tcga_slide_emb_df[tcga_slide_emb_df['file_name'].isin(my_tcga_slide_emb_list)]

import torch
vit256_features_exi = torch.load(exi_tcga_path + tcga_slide_emb_df2['file_name'].iloc[0])
vit256_features_new = torch.load(my_tcga_path + tcga_slide_emb_df2['file_name'].iloc[0])

vit256_features_exi2 = torch.load(exi_tcga_path + tcga_slide_emb_df2['file_name'].iloc[1])
vit256_features_new2 = torch.load(my_tcga_path + tcga_slide_emb_df2['file_name'].iloc[1])



df_process_gen = pd.read_csv('/mnt/disks/data_dir/data/output/process_list_autogen.csv')

file_path = '/mnt/disks/data_dir/data/output/patches/TCGA-A1-A0SI-01Z-00-DX1.AB717348-F964-4F29-BBE2-972B7C640432.h5'
with h5py.File(file_path, "r") as f:
    imgs_256 = f['imgs'][:]
    coords_256 = f['coords'][:]


file_path = '/mnt/disks/data_dir/data/output_4096/patches/TCGA-A1-A0SI-01Z-00-DX1.AB717348-F964-4F29-BBE2-972B7C640432.h5'
with h5py.File(file_path, "r") as f:
    imgs_4096 = f['imgs'][:]
    coords_4096 = f['coords'][:]



file_path = '/mnt/disks/data_dir/data/output/patches/TCGA-A1-A0SN-01Z-00-DX1.5E9B85AE-AFB7-41DC-8A1B-BD6DA39B6540.h5'
with h5py.File(file_path, "r") as f:
    imgs_256b = f['imgs'][:]
    coords_256b = f['coords'][:]

file_path = '/mnt/disks/data_dir/data/output_4096/patches/TCGA-A1-A0SN-01Z-00-DX1.5E9B85AE-AFB7-41DC-8A1B-BD6DA39B6540.h5'
with h5py.File(file_path, "r") as f:
    imgs_4096b = f['imgs'][:]
    coords_4096b = f['coords'][:]

file_path = '/mnt/disks/data_dir/data/output_4096_fp/patches/TCGA-A1-A0SN-01Z-00-DX1.5E9B85AE-AFB7-41DC-8A1B-BD6DA39B6540.h5'
with h5py.File(file_path, "r") as f:
    coords_4096c = f['coords'][:]



coords_256_list = []
for i in range(coords_256.shape[0]):
    coords_256_list.append((coords_256[i, 0], coords_256[i, 1]))

coords_4096_list = []
for i in range(coords_4096.shape[0]):
    coords_4096_list.append((coords_4096[i, 0], coords_4096[i, 1]))

coords_256_set = set(coords_256_list)
coords_4096_set = set(coords_4096_list)

df_256 = pd.DataFrame([{'top_x': i[0], 'top_y': i[1]} for i in coords_256_list])
df_4096 = pd.DataFrame([{'top_x': i[0], 'top_y': i[1]} for i in coords_4096_list])

df_256['bot_x'] = df_256['top_x'] + 256
df_256['bot_y'] = df_256['top_y'] + 256

df_4096['bot_x'] = df_4096['top_x'] + 4096
df_4096['bot_y'] = df_4096['top_y'] + 4096

data_list = []
for i, row in df_256.iterrows():

    cond_x_min = row.top_x >= df_4096['top_x']
    cond_x_max = row.bot_x <= df_4096['bot_x']
    cond_y_min = row.top_y >= df_4096['top_y']
    cond_y_max = row.bot_y <= df_4096['bot_y']

    df_4096_temp = df_4096[
        cond_x_min & cond_x_max & cond_y_min & cond_y_max
    ].reset_index(drop=True)
    if len(df_4096_temp) == 1:
        data_list.append({'top_x': df_4096_temp['top_x'].iloc[0], 'top_y': df_4096_temp['top_y'].iloc[0]})
    elif len(df_4096_temp) > 1:
        print('Trop')
        break
    else:
        print('None')
        data_list.append({'top_x': -1, 'top_y': -1})


df_256['ass_top_x'] = [x['top_x'] for x in data_list]
df_256['ass_top_y'] = [x['top_y'] for x in data_list]


file_path = '/mnt/disks/data_dir/data/output_256_fp/patches/TCGA-A1-A0SI-01Z-00-DX1.AB717348-F964-4F29-BBE2-972B7C640432.h5'
with h5py.File(file_path, "r") as f:
    coords_256 = f['coords'][:]

file_path = '/mnt/disks/data_dir/data/features_256_fp/h5_files/TCGA-A1-A0SI-01Z-00-DX1.AB717348-F964-4F29-BBE2-972B7C640432.h5'
with h5py.File(file_path, "r") as f:
    coords_256 = f['coords'][:]
    features_256 = f['features'][:]



file_path = '/mnt/disks/data_dir/data/features_4096_fp/h5_files/TCGA-A1-A0SI-01Z-00-DX1.AB717348-F964-4F29-BBE2-972B7C640432.h5'
with h5py.File(file_path, "r") as f:
    coords_4096 = f['coords'][:]
    features_4096 = f['features'][:]





# import shutil
#
# for i, row in df_svs.iterrows():
#     shutil.copy(data_path + row.folder + '/' + row.file_name, '/mnt/disks/data_dir/data/gdc_clean/')

import shutil

# for i, row in df_svs.iterrows():
#     if i < 500:
#         continue
#     else:
#         try:
#             os.remove('/mnt/disks/data_dir/data/gdc_clean/' + row.file_name)
#         except:
#             print(i)




# df_tsv = df[df['extension'] == 'tsv'].reset_index(drop=True)
#
#
# df_tsv_gene_counts = df_tsv[df_tsv['file_name'].map(lambda x: 'rna_seq.augmented_star_gene_counts.tsv' in x)].reset_index(drop=True)
# df_tsv_gene_level = df_tsv[df_tsv['file_name'].map(lambda x: 'gene_level_copy_number.v36.tsv' in x)].reset_index(drop=True)
# df_tsv_rppa = df_tsv[df_tsv['file_name'].map(lambda x: 'RPPA_data.tsv' in x)].reset_index(drop=True)
#
#
#
#
# out_path1 = '/mnt/disks/data_dir/data/gdc_clean/gene_counts/metadata/'
# out_path2 = '/mnt/disks/data_dir/data/gdc_clean/gene_counts/count_data/'
#
# prepare_dir(out_path1)
# prepare_dir(out_path2)
#
#
# data_list10 = []
#
# df_dic = {}
# for i, row in df_tsv_gene_counts.iterrows():
#     if i % 10 == 0:
#         print(i)
#     f_path = data_path + row.folder + '/' + row.file_name
#     try:
#         with open(f_path, 'r') as f:
#             lines = f.readlines()
#             gene_model = lines[0].strip()
#             line_2_clean = lines[2].strip()
#             line_3_clean = lines[3].strip()
#             line_4_clean = lines[4].strip()
#             line_5_clean = lines[5].strip()
#
#             data_list10.append({
#                 'folder': row.folder,
#                 'file_name': row.file_name,
#                 'gene_model': gene_model,
#                 'n_unmapped_1': int(line_2_clean.split('\t')[-3]),
#                 'n_unmapped_2': int(line_2_clean.split('\t')[-2]),
#                 'n_unmapped_3': int(line_2_clean.split('\t')[-1]),
#                 'n_multimapping_1': int(line_3_clean.split('\t')[-3]),
#                 'n_multimapping_2': int(line_3_clean.split('\t')[-2]),
#                 'n_multimapping_3': int(line_3_clean.split('\t')[-1]),
#                 'n_no_feature_1': int(line_4_clean.split('\t')[-3]),
#                 'n_no_feature_2': int(line_4_clean.split('\t')[-2]),
#                 'n_no_feature_3': int(line_4_clean.split('\t')[-1]),
#                 'n_ambiguous_1': int(line_5_clean.split('\t')[-3]),
#                 'n_ambiguous_2': int(line_5_clean.split('\t')[-2]),
#                 'n_ambiguous_3': int(line_5_clean.split('\t')[-1]),
#             })
#
#             data_list20 = []
#             for l in lines[6:]:
#                 line_split = l.split('\t')
#                 data_list20.append({
#                     'folder': row.folder,
#                     'file_name': row.file_name,
#                     'gene_id': line_split[0],
#                     'gene_name': line_split[1],
#                     'gene_type': line_split[2],
#                     'unstranded': line_split[3],
#                     'stranded_first': line_split[4],
#                     'stranded_second': line_split[5],
#                     'ttpm_unstranded': line_split[6],
#                     'tfpkm_unstranded': line_split[7],
#                     'tfpkm_uq_unstranded': line_split[8].strip(),
#                 })
#
#             temp_df = pd.DataFrame(data_list20)
#             temp_df.to_csv(out_path2 + row.file_name.split('.')[0] + '-gene_count.csv', index=False)
#     except:
#         print(f'File {f_path} failed!')
#
# temp_df2 = pd.DataFrame(data_list10)
# temp_df2.to_csv(out_path1 + 'gene_count_metadata.csv', index=False)


