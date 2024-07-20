#!usr/bin/env python
# -*- coding: utf-8 -*-

#Seccion 4
#Codigo informatico principal, para el analisis de datos

import time, sys, os, math
from random import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
from scipy.stats import norm
from scipy.stats import skellam
from scipy.stats import linregress
from sklearn.linear_model import LogisticRegression

#Funcion F_X logistica habitual en el sistema Elo
def l_cdf(x):
	if x>20000:
		return 1.
	if x<-20000:
		return 0.
	return 1 - 1/(1+10**(x/400.))

#Funcion F_X normal con varianza 200
def n_cdf(x):
	return norm.cdf(x/200.)

#Funcion F_X del modelo discreto, basada en la distr. skellam
global A
A = 2.57
def sk_cdf(x):
	mu1 = x/2. + (x*x/4.+A*A)**0.5
	mu2 = -x/2. + (x*x/4.+A*A)**0.5
	return skellam.cdf(-1, mu2, mu1) + 0.5*skellam.pmf(0,mu1,mu2)

def sk_linear_cdf(x):
	mu1 = (A+x)/2.
	mu2 = (A-x)/2.
	return skellam.cdf(-1, mu2, mu1) + 0.5*skellam.pmf(0,mu1,mu2)

def sk_sqrt_cdf(x):
	mu1 = ((A+x/A)/2.)**2
	mu2 = ((A-x/A)/2.)**2
	return skellam.cdf(-1, mu2, mu1) + 0.5*skellam.pmf(0,mu1,mu2)

#Funcion F_X para X uniforme en un intervalo (relacionada con el sis. Harkness)
def sq_cdf(x):
	if x<-500:
		return 0.0
	if x>500:
		return 1.0
	return x/1000.0+0.5

#Funcion de comparacion del marcador p_A(G_A, G_B)
def score_f(result):
	return (int(result[0])>int(result[1]))+0.5*(result[0]==result[1])

default_cdf = n_cdf

#Minimo de una funcion de una variable por el metodo de busqueda de la seccion aurea
def minimo(f, intervalo, tolerancia = 0.1, fa = None, fb = None):
	phi = 0.5*(1+5.**0.5)
	a, b = intervalo
	c = a + (b-a)/phi
	evals = 1
	if fa is None:
		fa, evals = f(a), evals+1
	if fb is None:
		fb, evals = f(b), evals+1
	fc = f(c)
	while abs(b-a) > tolerancia:
		d = a + (c-a)/phi
		fd = f(d)
		#print( d,fd)
		evals += 1
		if fd < fc:
			a, b, c = a, c, d
			fa, fb, fc = fa, fc, fd
		else:
			a, b, c = b, d, c
			fa, fb, fc = fb, fd, fc
	return c, fc, evals

def line_search(f, x, grad, tolerance = 1e-6):
	f1 = fa = f(x)
	f2 = f(x+grad)
	evals = 2
	mult = 1
	while f1 > f2:
		mult = 3*mult
		f1, f2 = f2, f(x+mult*grad)
		evals += 1
	alpha, line_min, _evals = minimo(lambda y: f(x+y*grad), [0, mult], tolerancia = tolerance, fa=fa, fb=f2)
	return alpha, line_min, evals+_evals


def conjugate_gradient_descent(f, x, descent = None, tolerance = 1e-6, max_iter = 10):
	# Minimizes f starting at x, "descent" returns a descent direction such as the gradient
	descent_cost = [0, 1]
	if descent is None:
		# if no function is provided, we approximate the gradient
		def gradient(xx):
			grad = []
			for ii in range(len(xx)):
				xx_eps = xx.copy()
				xx_eps[ii] += 0.5*tolerance
				xx_meps = xx.copy()
				xx_meps[ii] -= 0.5*tolerance
				grad.append((f(xx_eps) - f(xx_meps))/tolerance)
			# print('Gradient', grad)
			return -np.array(grad)
		descent = gradient
		descent_cost = [2*len(x), 0] # cost in evaluation of getting a descent direction
	r = descent(x)
	evaluations = descent_cost.copy()
	p = r.copy()
	# residue_norms = []
	f_x = 0
	for iters in range(max_iter):
		# residue_norms.append(np.matmul(r, r)**0.5)
		# if residue_norms[-1] < tolerance:
		# 	break
		norm_p = np.matmul(p, p)**0.5
		alpha, f_x, evals = line_search(f, x, p, tolerance = tolerance/norm_p)
		evaluations[0] += evals
		x = x + alpha*p
		# print ('NLCG', x)
		if alpha*norm_p < tolerance:
			break
		new_r = descent(x)
		evaluations = [evaluations[0]+descent_cost[0], evaluations[1]+descent_cost[1]]
		beta = np.matmul(new_r, new_r)/np.matmul(r, r)
		r = new_r
		p = r + beta*p
	else:
		return x, f_x, iters, 1  # Flag 1 --> Maximum iterations reached
	print ('NLCGD evals:', evaluations)
	return x, f_x, iters, 0  # Flag 0 --> Solved correctly


#Implementacion de la funcion definida en el modelo dinamico
#para hallar la esperanza de la puntuacion p
def stoc_cdf(e1,e2,score,u_t, distr = default_cdf, damp=1, c_factor = 150, l_factor = 0):
	return distr((l_factor+(e1-e2)*damp)*(u_t**0.5) + c_factor*score/(u_t**0.5+1e-6))
	# return distr((l_factor+(e1-e2)*damp)*(((t+0.0)/tiempo_total)**0.5) + c_factor*score/((t+0.01)/tiempo_total)**0.5)

#Calculo de los errores epsilon(M,R)
def error(jugadores, ratings, resultados, cdf = default_cdf, l_factor = 0):
	errores = [0. for jj in jugadores]
	for j1, j2, puntos in resultados:
		index1 = jugadores.index(j1)
		index2 = jugadores.index(j2)
		aux = cdf(l_factor + ratings[index1] - ratings[index2])
		errores[index1] += puntos - aux
		errores[index2] += aux - puntos
	return errores

#Calculo del estimador r que se estudia en el anexo 1
def performance(jugadores, elos, resultados, cdf = default_cdf, k = 25., default = 2000., l_factor=0):
	if cdf == sk_cdf and k==25.0:
		k = 0.6
	if cdf == sk_sqrt_cdf and k==25.0:
		k = .2
	variable = [False for jj in jugadores]
	ratings = [default for jj in jugadores]
	for ii in range(len(jugadores)):
		if elos[ii] == "-1":
			variable[ii] = True
		else:
			ratings[ii] = elos[ii]
	for cc in range(1, 30):
		errores = error(jugadores, ratings, resultados, cdf = cdf, l_factor = l_factor)
		error_total = sum([(errores[ii]**2)*variable[ii] for ii in range(len(jugadores))])
		# print( 'e', error_total)
		if error_total < 1e-9:
			break
		for ii in range(len(ratings)):
			if variable[ii]:
				ratings[ii] += k*errores[ii]
	for cc in range(1, 100):
		errores = error(jugadores, ratings, resultados, cdf = cdf, l_factor = l_factor)
		error_total = sum([(errores[ii]**2)*variable[ii] for ii in range(len(jugadores))])
		# print( error_total)
		if error_total < 1e-9:
			break
		for ii in range(len(ratings)):
			if variable[ii]:
				ratings[ii] += k*errores[ii]
		errores = error(jugadores, ratings, resultados, cdf = cdf, l_factor = l_factor)
		error_total = sum([(errores[ii]**2)*variable[ii] for ii in range(len(jugadores))])
		# print( 'E', error_total)
		if error_total < 1e-9:
			break
		derivadas = [0.]*len(jugadores)
		for jj in range(len(ratings)):
			if variable[jj]:
				derivadas[jj] = error(jugadores, ratings[:jj] + [ratings[jj] + 0.0001] + ratings[jj+1:], resultados, cdf = cdf, l_factor = l_factor)[jj]
				derivadas[jj] -= error(jugadores, ratings[:jj] + [ratings[jj] - 0.0001] + ratings[jj+1:], resultados, cdf = cdf, l_factor = l_factor)[jj]
		for jj in range(len(ratings)):
			if variable[jj] and derivadas[jj] != 0.0:
				ratings[jj] -= 0.0002*errores[jj]/derivadas[jj]
	else:
		# print( "Bajando +1")
		return performance(jugadores, elos, resultados, cdf = cdf, k=k/2., default = default, l_factor = l_factor)
	return ratings

#Extraccion de una base de datos de partidos evaluados (con rating) 
#a partir de una muestra de partidos sin evaluar
#implementando el algoritmo descrito en el capitulo 4
def rated_db(dbfile, cdf, score_function = score_f, k_factor = 20, muestra = 15, l_factor = 0, mostrar = False):
	t0 = time.time()
	arch = open(dbfile, 'r')
	lineas = [el.split("\t") for el in arch.read().split("\n")[:-1]]
	arch.close()
	temporadas = sorted(list(set([el[0] for el in lineas])))
	equipos = sorted(list(set([el[2] for el in lineas])))
	eloEquipos = {}
	nEquipos = {}
	for eq in equipos:
		nEquipos[eq] = 0
		eloEquipos[eq] = "-1"
	rated_games = []
	sample_games = []
	temp_actual = "-1"
	for line in lineas:
		temp = line[0]
		if temp != temp_actual:
			sample_games = []
			temp_actual = temp
			aux_teams = [line[2:4] for line in lineas if line[0] == temp]
			equipos_temp = list(set([teams[0] for teams in aux_teams] + [teams[1] for teams in aux_teams]))
			for eq in equipos: # Reset rating of teams not playing this season
				if eq not in equipos_temp:
					eloEquipos[eq] = "-1"
					nEquipos[eq] = 0
		nEquipos[line[2]] += 1
		nEquipos[line[3]] += 1
		if eloEquipos[line[2]] == "-1" or eloEquipos[line[3]] == "-1":
			sample_games.append([line[2], line[3], score_function(line[4:6])])
			check = True
			for eq in equipos_temp: # We don't assign ratings to new teams until they have played enough games
				if nEquipos[eq] < muestra:
					check = False
					break
			if check:
				#print( "Haciendo pfm", temp_actual)
				#print( [eloEquipos[eq] for eq in equipos_temp])
				lis = performance(equipos_temp, [eloEquipos[eq] for eq in equipos_temp], sample_games, cdf = cdf, l_factor = l_factor)
				for ii in range(len(equipos_temp)):
					eloEquipos[equipos_temp[ii]] = lis[ii]
			continue
		equipo1 = line[2]
		equipo2 = line[3]
		result = line[4:6]
		e_score = cdf(l_factor + eloEquipos[equipo1]-eloEquipos[equipo2])
		score = score_function(result)
		rated_games.append(line[:4] + [score, eloEquipos[equipo1], eloEquipos[equipo2], line[6:]])
		eloEquipos[equipo1] += (score-e_score)*k_factor
		eloEquipos[equipo2] -= (score-e_score)*k_factor
	if mostrar:
		print( "%s evaluada en %.2f segundos"%(dbfile, time.time()-t0))
		print( "%d partidos extraidos de %d ()"%(len(rated_games), len(lineas)))#, 100.*len(rated_games)/len(lineas))
	return rated_games

def adjusted_goals(off1, def1, off2, def2, goals1, goals2, goal_avg_home, goal_avg_away):
	# Used for the calculation of SPI ratings
	goal_avg = 0.5*(goal_avg_away + goal_avg_home)
	ags1 = (goals1 - def2)*(goal_avg*0.424+0.548)/max(0.25, def2*0.424+0.548) + goal_avg_away
	aga1 = (goals2 - off2)*(goal_avg*0.424+0.548)/max(0.25, off2*0.424+0.548) + goal_avg_home
	ags2 = (goals2 - def1)*(goal_avg*0.424+0.548)/max(0.25, def1*0.424+0.548) + goal_avg_home
	aga2 = (goals1 - off1)*(goal_avg*0.424+0.548)/max(0.25, off1*0.424+0.548) + goal_avg_away
	return ags1, aga1, ags2, aga2

def expected_score_spi(off1, def1, off2, def2, goal_avg_home, goal_avg_away):
	avg_total = 0.5*(goal_avg_home + goal_avg_away)
	expected_score1 = (off1-goal_avg_away)*max(0.25, def2*0.424+0.548)/(avg_total*0.424+0.548) + def2
	expected_take1 = (def1-goal_avg_home)*max(0.25, off2*0.424+0.548)/(avg_total*0.424+0.548) + off2
	expected_score2 = (off2-goal_avg_home)*max(0.25, def1*0.424+0.548)/(avg_total*0.424+0.548) + def1
	expected_take2 = (def2-goal_avg_away)*max(0.25, off1*0.424+0.548)/(avg_total*0.424+0.548) + off1
	# print (expected_score1, expected_take1, expected_score2, expected_take2)
	expected_goals1 = 0.5*(expected_score1+expected_take2)
	expected_goals2 = 0.5*(expected_score2+expected_take1)
	if expected_goals1 < 0:
		# print ('expected_goals1 < 0:', expected_goals1)
		expected_goals1 = 1e-6
	if expected_goals2 < 0:
		# print ('expected_goals2 < 0:', expected_goals2)
		expected_goals2 = 1e-6
	return expected_goals1, expected_goals2, skellam.cdf(-1, expected_goals2, expected_goals1) + 0.5*skellam.pmf(0,expected_goals1,expected_goals2)

#Estimator for SPI ratings based in a sample of unrated games
def spi_performance(jugadores, off_rating, def_rating, matches, lambda_factor, aghome, agaway):
	variable = [False for jj in jugadores]
	default = 0.5*(aghome + agaway)
	offensive = [default for jj in jugadores]
	defensive = [default for jj in jugadores]
	for ii in range(len(jugadores)):
		if off_rating[ii] == "-1":
			variable[ii] = True
		else:
			offensive[ii] = off_rating[ii]
			defensive[ii] = def_rating[ii]
	for cc in range(1, 20):
		off_cp = offensive.copy()
		def_cp = defensive.copy()
		for team1, team2, goals1, goals2 in matches:
			index1, index2 = jugadores.index(team1), jugadores.index(team2)
			ags1, aga1, ags2, aga2 = adjusted_goals(offensive[index1], defensive[index1], offensive[index2], defensive[index2], goals1, goals2, aghome, agaway)
			if variable[index1]:
				offensive[index1] += lambda_factor*(ags1 - offensive[index1])
				defensive[index1] += lambda_factor*(aga1 - defensive[index1])
			if variable[index2]:
				offensive[index2] += lambda_factor*(ags2 - offensive[index2])
				defensive[index2] += lambda_factor*(aga2 - defensive[index2])
		# print(sum([abs(a - b) for a,b in zip(offensive, off_cp)]))
		# print(sum([abs(a - b) for a,b in zip(defensive, def_cp)]))
	return offensive, defensive

def spi_rated_db(dbfile, score_function = score_f, lambda_factor = 0.1, muestra = 12, mostrar = False):
	t0 = time.time()
	arch = open(dbfile, 'r')
	lineas = [el.split("\t") for el in arch.read().split("\n")[:-1]]
	arch.close()
	temporadas = sorted(list(set([el[0] for el in lineas])))
	equipos = sorted(list(set([el[2] for el in lineas])))
	offEquipos = {} # Offensive rating
	defEquipos = {} # Defensive rating
	nEquipos = {}
	avg_goals_home = sum(int(el[4]) for el in lineas)/len(lineas)
	avg_goals_away = sum(int(el[5]) for el in lineas)/len(lineas)
	# print('AVG goals home/away:', avg_goals_home, avg_goals_away)
	for eq in equipos:
		nEquipos[eq] = 0
		offEquipos[eq] = "-1"
		defEquipos[eq] = "-1"
	rated_games = []
	sample_games = []
	temp_actual = "-1"
	for line in lineas:
		temp = line[0]
		if temp != temp_actual:
			sample_games = []
			temp_actual = temp
			aux_teams = [line[2:4] for line in lineas if line[0] == temp]
			equipos_temp = list(set([teams[0] for teams in aux_teams] + [teams[1] for teams in aux_teams]))
			for eq in equipos: # Reset rating of teams not playing this season
				if eq not in equipos_temp:
					offEquipos[eq] = "-1"
					defEquipos[eq] = "-1"
					nEquipos[eq] = 0
		nEquipos[line[2]] += 1
		nEquipos[line[3]] += 1
		if defEquipos[line[2]] == "-1" or defEquipos[line[3]] == "-1":
			sample_games.append([line[2], line[3], int(line[4]), int(line[5])])
			check = True
			for eq in equipos_temp: # We don't assign ratings to new teams until they have played enough games
				if nEquipos[eq] < muestra:
					check = False
					break
			if check:
				#print( "Haciendo pfm", temp_actual)
				#print( [eloEquipos[eq] for eq in equipos_temp])
				off_rating, def_rating = [offEquipos[eq] for eq in equipos_temp], [defEquipos[eq] for eq in equipos_temp]
				off_rating, def_rating = spi_performance(equipos_temp, off_rating, def_rating, sample_games, lambda_factor, avg_goals_home, avg_goals_away)
				for ii in range(len(equipos_temp)):
					# print (equipos_temp[ii], off_rating[ii], def_rating[ii])
					offEquipos[equipos_temp[ii]] = off_rating[ii]
					defEquipos[equipos_temp[ii]] = def_rating[ii]
			continue
		team1 = line[2]
		team2 = line[3]
		result = line[4:6]
		# e_score = cdf(l_factor + eloEquipos[team1]-eloEquipos[team2])
		score = score_function(result)
		ags1, aga1, ags2, aga2 = adjusted_goals(offEquipos[team1], defEquipos[team1], offEquipos[team2], defEquipos[team2], int(result[0]), int(result[1]), avg_goals_home, avg_goals_away)
		rated_games.append(line[:4] + [score, offEquipos[team1], defEquipos[team1], offEquipos[team2], defEquipos[team2], line[6:]])
		offEquipos[team1] += (ags1-offEquipos[team1])*lambda_factor
		defEquipos[team1] += (aga1-defEquipos[team1])*lambda_factor
		offEquipos[team2] += (ags2-offEquipos[team2])*lambda_factor
		defEquipos[team2] += (aga2-defEquipos[team2])*lambda_factor
	if mostrar:
		print( "%s evaluada en %.2f segundos"%(dbfile, time.time()-t0))
		print( "%d partidos extraidos de %d ()"%(len(rated_games), len(lineas)))#, 100.*len(rated_games)/len(lineas))
	return rated_games, avg_goals_home, avg_goals_away

def spi_mse(dbfile, lambda_factor = 0.1, muestra = 12, damp = 1):
	rated_games, avg_goals_home, avg_goals_away = spi_rated_db(dbfile, lambda_factor = lambda_factor, muestra = muestra)
	mse = 0.0
	for season,date,t1,t2,res,off1,def1,off2,def2,hist in rated_games:
		g1, g2, exp_score = expected_score_spi(off1*damp, def1*damp, off2*damp, def2*damp, avg_goals_home, avg_goals_away)
		mse += (exp_score - res)*(exp_score - res)
	# print (lambda_factor, damp, mse/len(rated_games))
	return mse/len(rated_games)


def different_mean_test(list1, list2):
	list3 = [yy-xx for xx, yy in zip(list1, list2)]
	return norm.cdf(sum(list3)/np.std(list3)/(len(list3)**0.5))
	# mean1, mean2 = sum(list1)/len(list1), sum(list2)/len(list2)
	# var1, var2 = np.std(list1)**2, np.std(list2)**2
	# print ('m1, m2, v1, v2', mean1, mean2, var1, var2)
	# return norm.cdf((mean1-mean2)/((var1/len(list1)+var2/len(list2))**0.5))

#Clase "base de datos", que opera con la lista de partidos
#que proporciona la funcion rated_db
class Db:
	def __init__(self, rated_games):
		self.games = rated_games
	#Funciones del error cuadratico de los residuos
	def scr(self, cdf, l_factor = 0, damp = 1):
		scr = 0.0
		for a,b,c,d,s,e1,e2,g in self.games:
			scr += (s-cdf(l_factor+damp*(e1-e2)))**2
		return scr
	def mcr(self, cdf, l_factor = 0, damp = 1):
		return self.scr(cdf, l_factor=l_factor, damp=damp)/len(self.games)
	def all_sq_error(self, cdf, l_factor = 0, damp = 1):
		return [(s-cdf(l_factor+damp*(e1-e2)))**2 for a,b,c,d,s,e1,e2,g in self.games]

	#Funcion del error cuadratico total (sin usar sist. Elo)
	def sct(self, cdf, l_factor=0):
		sct = 0.0
		x = cdf(l_factor)
		for a,b,c,d,s,e1,e2,g in self.games:
			sct += (s-x)**2
		return sct

	#Curva ROC de la base de datos
	def curva_roc(self, cdf, l_factor=0, color = "b"):
		datos = []
		for x1,x2,x3,x4,p,r1,r2,goles in self.games:
			if p == 1.0 or p == 0.0:
				esperanza = cdf(l_factor+r1-r2)
				datos.append((esperanza, p))
		datos = sorted(datos, key = lambda x: x[0])
		datos = [x[1] for x in datos]
		positivos = sum(datos)
		negativos = len(datos)-positivos
		false_neg = 0.
		true_neg = 0.
		curva = [(1.,1.)]
		for valor in datos:
			if valor:
				false_neg += 1
			else:
				true_neg += 1
			curva.append((1 - true_neg/negativos, 1 - false_neg/positivos))
		plt.plot([c[0] for c in curva], [c[1] for c in curva], color)
		plt.plot([0,1],[0,1], "r")
		#plt.show()

	def result_vs_rating_diff(self, cdf, l_factor = 0, n_points = 20, show = False):
		sorted_games = sorted(self.games, key = lambda x: x[-3]-x[-2])
		sample_rating_diff = []
		sample_result = []
		for ii in range(n_points):
			quantile1, quantile2 = int(ii*len(sorted_games)/n_points), int((ii+1)*len(sorted_games)/n_points)
			avg_result = 0.0
			avg_rating_diff = 0.0
			for x1, x2, x3, x4, p, r1, r2, goals in sorted_games[quantile1:quantile2]:
				avg_result += p
				avg_rating_diff += r1-r2
			sample_result.append(avg_result/(quantile2-quantile1))
			sample_rating_diff.append(avg_rating_diff/(quantile2-quantile1))
		plt.scatter(sample_rating_diff, sample_result, label='Sample Games', zorder=2)
		x_range = [sample_rating_diff[0] + (sample_rating_diff[-1]-sample_rating_diff[0])*ii/1000 for ii in range(1001)]
		plt.plot(x_range, [cdf(xx+l_factor) for xx in x_range], label='D=1', zorder=1)
		# plt.plot(x_range, [cdf(0.8*xx+0.8*l_factor) for xx in x_range])
		plt.plot(x_range, [cdf(0.8*xx+l_factor) for xx in x_range], label='D=0.8', zorder=1) ######
		plt.legend()
		plt.xlabel('Rating difference')
		plt.ylabel('Result')
		if show:
			plt.show()
		else:
			plt.savefig('out/result_vs_rating.png', bbox_inches='tight')
		plt.clf()


	#Curva del error cuadratico medio de p en funcion del tiempo
	def mcr_dinamico(self, predictor, cdf, l_factor=0, damp=1, c_factor = 0, t = 90, intervalo = 1, u_t = None):
		scr = [0.0]*(t+1)
		if u_t is None:
			u_function = lambda x: (t-x)/t
		else:
			u_function = lambda x: u_t[x]
		for a,b,c,d,s,e1,e2,g in self.games:
			score = 0
			jj = 0
			for ii in range(0,t,intervalo):
				if jj<len(g):
					tiempo, res = g[jj].split(";")
					while int(tiempo)<=ii:
						s1,s2 = map(int, res.split("-"))
						score = s1-s2
						jj+=1
						if jj >= len(g):
							break
						tiempo, res = g[jj].split(";")
				scr[ii] += (s-predictor(e1,e2,score, u_function(ii), distr = cdf, l_factor = l_factor, damp=damp, c_factor = c_factor))**2
		return [scr[ii]/len(self.games) for ii in range(0,t,intervalo)]

	#Tests de normalidad con los residuos de la diferencia del marcador
	def test_normal(self, l_factor, c_factor, damp=1, deporte = "X"):
		residuos = []
		dif_elos = []
		for a,b,c,d,p,e1,e2,g in self.games:
			s1,s2 = g[-1].split(";")[1].split("-")
			dif_elos.append(e1-e2)
			residuos.append(c_factor*(int(s1)-int(s2)) - (l_factor + damp*(e1 - e2)) )
		n = len(self.games)
		gradient,intercept,r_value,p_value,std_err = grafico_reg_lineal(dif_elos, residuos, 'out/test_norm_5_%s.png'%deporte, modo='scatter')
		print ('Residuals correlation p_value = ', p_value, std_err)
		print ('Residuals STD = ', np.std(residuos))
		plt.scatter(range(n), residuos, s = 1)
		#plt.show()
		plt.savefig('out/test_norm_1_%s.png'%deporte, bbox_inches='tight')
		plt.clf()
		ind = sorted(range(len(self.games)), key = lambda x: dif_elos[x])
		residuos = [residuos[ind[ii]] for ii in range(n)]
		dif_elos = sorted(dif_elos)
		plt.scatter(dif_elos, residuos, s = 1)
		#plt.show()
		plt.savefig('out/test_norm_2_%s.png'%deporte, bbox_inches='tight')
		plt.clf()
		residuos = sorted(residuos)
		plt.scatter([norm.ppf((x+1.)/(n+1)) for x in range(n)], residuos, s = 1)
		plt.savefig('out/test_norm_3_%s.png'%deporte, bbox_inches='tight')
		plt.clf()
		grafico_reg_lineal([norm.ppf((x+1.)/(n+1)) for x in range(n)], residuos, 'out/test_norm_4_%s.png'%deporte, modo='scatter')

	#Calculo de los caminos B_t en el intervalo [0,1]
	#reconstruidos a partir de una muestra aleatoria de partidos
	def caminos_est(self, l_factor, c_factor, muestra=10, t=90, deporte = "X"):
		lis = []
		n = len(self.games)
		while len(lis)<muestra:
			rr = int(n*random())
			if rr not in lis:
				lis.append(rr)
		caminos = []
		for rr in lis:
			caminos.append([])
			a,b,c,d,s,e1,e2,g = self.games[rr]
			score = 0
			jj = 0
			for ii in range(t):
				if jj<len(g):
					tiempo, res = g[jj].split(";")
					while int(tiempo)<=ii:
						s1,s2 = map(int, res.split("-"))
						score = s1-s2
						jj+=1
						if jj >= len(g):
							break
						tiempo, res = g[jj].split(";")
				caminos[-1].append((c_factor*score-(l_factor+e1-e2)*ii/t)/200.)
			plt.plot([x/(t+0.0) for x in range(t)], caminos[-1])
		#plt.show()
		plt.savefig('out/caminos_%s.png'%deporte, bbox_inches='tight')
		plt.clf()
		return caminos
	
	#Curva del error cuadratico medio de S_T en funcion del tiempo
	def test_sesgo_score(self, l_factor, c_factor, t = 90, deporte = "X", u_t=None):
		scr = [0.0]*(t+1)
		u_function = lambda x: (t - x)/t
		if u_t is not None:
			u_function = lambda x: u_t[x]
		for a,b,c,d,s,e1,e2,g in self.games:
			score = 0
			jj = 0
			s1, s2 = g[-1].split(";")[1].split("-")
			s_final = int(s1)-int(s2)
			for ii in range(0,t):
				if jj<len(g):
					tiempo, res = g[jj].split(";")
					while int(tiempo)<=ii:
						s1,s2 = map(int, res.split("-"))
						score = s1-s2
						jj+=1
						if jj >= len(g):
							break
						tiempo, res = g[jj].split(";")
				scr[ii] += (s_final-score-u_function(ii)*(l_factor+e1-e2)/c_factor)**2
		scr = [item/len(self.games) for item in scr]
		x_data = range(0,t+1)
		if u_t is not None:
			x_data = [1-u_t[xx] for xx in x_data]
		plt.plot(x_data, scr)
		#plt.show()
		plt.savefig('out/test_sesgo_%s.png'%deporte, bbox_inches='tight')
		plt.clf()
		grafico_reg_lineal(x_data, scr, 'out/ecm_score_%s.png'%deporte)

def time_analysis(filenames, rounding = 1, scaling = 1, saveplt = None):
	games = []
	for filename in filenames:
		with open(filename, 'r') as arch:
			lines = [el.split('\t') for el in arch.read().split('\n')][:-1]
			games = games + lines
	times = {}
	increase = {}
	for game in games:
		# print(game)
		score_a, score_b = 0,0
		for goal_time in game[game.index('0;0-0')+1:]:
			time, score = goal_time.split(';')
			time = int(time)//rounding
			ii, jj = int(score.split('-')[0]), int(score.split('-')[1])
			if time not in times:
				times[time] = 0
				increase[time] = 0
			times[time] += 1
			increase[time] += ii-score_a + jj-score_b
			score_a, score_b = ii, jj
	for key in times:
		times[key] = times[key]/len(games)
		increase[key] = increase[key]/len(games)
	max_time = max([key for key in times])
	# plt.bar([tt*rounding/scaling for tt in range(max_time+1)], [0 if tt not in times else times[tt] for tt in range(max_time+1)], width=rounding/scaling)
	# plt.show()
	# plt.clf()
	plt.bar([tt*rounding/scaling for tt in range(max_time+1)], [0 if tt not in times else increase[tt] for tt in range(max_time+1)], width=rounding/scaling)
	plt.xlabel('Game Time (minutes)')
	plt.ylabel('Average points scored')
	if saveplt is None:
		plt.autoscale(enable=True, axis='x', tight=True)
		plt.show()
	else:
		plt.savefig(saveplt, bbox_inches='tight')
	plt.clf()
	# print(times)
	u_t = {}
	suma = 0.0
	total = sum(increase[ii] for ii in increase)
	for time in range(max_time+1):
		suma += 0.0 if time not in increase else increase[time]
		u_t[time] = 1-suma/total
	# print(max_time)
	return u_t

#Elaboracion de todo el analisis discreto de una base de datos,
#encontrando los valores optimos de K y L
def analisis_estatico(archivo, archivo_test, f_X, inter_k = [0,40], inter_l = [-150, 150], muestra = 12, mostrar = False, deporte = "X", tol_1 = 0.01, tol_2 = 1e-3):
	print( "\n{*} Analisis estatico:", deporte)
	table_anova = []
	error_mcr = lambda x, y, z: Db(rated_db(archivo, f_X, k_factor = x, l_factor = y, muestra = muestra)).mcr(f_X, l_factor = y, damp=z)
	#Minimo en K con L=0
	mejor_k, mejor_mcr, _ = minimo(lambda x: error_mcr(x,0,1), inter_k, tolerancia = tol_1)
	print( "Fijado L=0 y minimizando en K:")
	print( "K=%.3f, MCR=%.5f"%(mejor_k, mejor_mcr))
	db1 = Db(rated_db(archivo_test, f_X, k_factor = mejor_k, l_factor = 0, muestra = muestra))
	mse_test = db1.mcr(f_X, l_factor = 0)
	print( "MCR_test=%.5f"%mse_test)
	table_anova.append([mejor_k, 0, 1, mejor_mcr, mse_test])
	Db(rated_db(archivo, f_X, k_factor = mejor_k, l_factor = 0, muestra = muestra)).curva_roc(f_X, l_factor = 0, color = "b")
	#Minimo en L con K=0
	mejor_l, mejor_mcr, _ = minimo(lambda y: error_mcr(0,y,1), inter_l, tolerancia = tol_1)
	print( "Fijado K=0 y minimizando en L:")
	print( "L=%.3f, MCR=%.5f"%(mejor_l, mejor_mcr))
	db2 = Db(rated_db(archivo_test, f_X, k_factor = 0, l_factor = mejor_l, muestra = muestra))
	mse_test = db2.mcr(f_X, l_factor = mejor_l)
	print( "MCR_test=%.5f"%mse_test)
	table_anova = [[0, mejor_l, 1, mejor_mcr, mse_test]] + table_anova
	mejor_l_2 = mejor_l
	Db(rated_db(archivo, f_X, k_factor = 0, l_factor = mejor_l, muestra = muestra)).curva_roc(f_X, l_factor = mejor_l, color = "g")
	#Minimo en K y L
	optimal, mejor_mcr, iters, flag = conjugate_gradient_descent(lambda x: error_mcr(x[0], x[1],1), np.array([mejor_k, mejor_l]), tolerance=tol_2)
	mejor_k, mejor_l = optimal
	print( "Minimo MCR en K y L:")
	print( "L=%.3f, K=%.3f, MCR=%.5f"%(mejor_l, mejor_k, mejor_mcr))
	test_db = Db(rated_db(archivo_test, f_X, k_factor = mejor_k, l_factor = mejor_l, muestra = muestra, mostrar = True))
	print( "MCR_test=%.5f"%test_db.mcr(f_X, l_factor = mejor_l))
	list_sq_error_1 = test_db.all_sq_error(f_X, l_factor = mejor_l)
	table_anova.append([mejor_k, mejor_l, 1, mejor_mcr, test_db.mcr(f_X, l_factor = mejor_l)])
	mydb = Db(rated_db(archivo, f_X, k_factor = mejor_k, l_factor = mejor_l, muestra = muestra, mostrar = True))
	mydb.curva_roc(f_X, l_factor = mejor_l, color = "k")
	azul = mpatches.Patch(color='blue', label='Fixing L=0')
	verde = mpatches.Patch(color='green', label='Fixing K=0')
	negro = mpatches.Patch(color='black', label='Global minimum')
	rojo = mpatches.Patch(color="red", label='y=x')
	plt.legend(handles=[verde,azul,negro,rojo])
	plt.title('Curva ROC')
	if mostrar:
		plt.show()
	else:
		plt.savefig('out/roc_%s.png'%deporte, bbox_inches='tight')
		plt.clf()
	db1.curva_roc(f_X, l_factor = 0, color = "b")
	db2.curva_roc(f_X, l_factor = mejor_l_2, color = "g")
	test_db.curva_roc(f_X, l_factor = mejor_l, color = "k")
	plt.legend(handles=[verde,azul,negro,rojo])
	plt.title('Curva ROC')
	if mostrar:
		plt.show()
	else:
		plt.savefig('out/roc_%s_test.png'%deporte, bbox_inches='tight')
		plt.clf()
	mejor_l2, mejor_sct, _ = minimo(lambda x: mydb.sct(f_X, l_factor = x), inter_l, tolerancia = tol_1)
	print( "Minimo MCT:")
	print( "L=%.3f, P=%.2f%%, MCR=%.5f"%(mejor_l2, 100*f_X(mejor_l2), mejor_sct/len(mydb.games)))
	table_anova = [["-----", mejor_l2, "-----", mejor_sct/len(mydb.games)]] + table_anova
	mejor_l2, mejor_sct, _ = minimo(lambda x: test_db.sct(f_X, l_factor = x), inter_l, tolerancia = tol_1)
	print( "Minimo MCT en test:")
	print( "L=%.3f, P=%.2f%%, MCR=%.5f"%(mejor_l2, 100*f_X(mejor_l2), mejor_sct/len(test_db.games)))
	test_db.mcr(f_X, l_factor = mejor_l)
	table_anova[0].append(mejor_sct/len(test_db.games))
	# Damping
	optimal, mejor_mcr, iters, flag = conjugate_gradient_descent(lambda x: error_mcr(x[0], x[1], x[2]), np.array([mejor_k, mejor_l, 1]), tolerance=tol_2)
	mejor_k, mejor_l, dampening = optimal
	print( "Minimo MCR en K, L & D:")
	print( "L=%.3f, K=%.3f, D=%.3f MCR=%.5f"%(mejor_l, mejor_k, dampening, mejor_mcr))
	test_db = Db(rated_db(archivo_test, f_X, k_factor = mejor_k, l_factor = mejor_l, muestra = muestra, mostrar = True))
	list_sq_error_2 = test_db.all_sq_error(f_X, l_factor = mejor_l, damp=dampening)
	mse_test = test_db.mcr(f_X, l_factor = mejor_l, damp = dampening)
	print( "MCR_test=%.5f"%mse_test)
	print ("p_value Dampening = ", different_mean_test(list_sq_error_1, list_sq_error_2))
	table_anova.append([mejor_k, mejor_l, dampening, mejor_mcr, mse_test])
	for it in [['K', 'L', 'Damp', 'MSE', 'MSEtest']] + table_anova:
		print(' | '.join(['%7.7s'%str(el) for el in it]))
	return mydb, mejor_k, mejor_l, dampening

#Elaboracion de todo el analisis del modelo discreto,
#y busqueda del valor optimo de C
def analisis_dinamico(mydb, db_test, f_X, l_factor, inter_c = [0,300], damp=1, t = 90, intervalo = 9, u_t = None, mostrar = False, unidades = "minutes", deporte = "X"):	
	print( "\n{*} Analisis dinamico:", deporte)
	media = lambda v: sum(v)/len(v)
	error_mcr = lambda x: media(mydb.mcr_dinamico(stoc_cdf, f_X, l_factor = l_factor, damp=damp, c_factor = x, t = t, intervalo = intervalo, u_t = u_t))
	mejor_c, mejor_mcr_dinamico, _ = minimo(error_mcr, inter_c, tolerancia = 1.)#0.01)#
	freq_output = max(1,int(t/100.))
	mcr_lista = mydb.mcr_dinamico(stoc_cdf, f_X, l_factor = l_factor, c_factor = mejor_c, t = t, intervalo = freq_output, u_t = u_t)
	# mejor_mcr_dinamico /= len(mcr_lista)
	print( "Minimo MCR dinamico:")
	print( "C=%.2f, MCR=%.5f"%(mejor_c, mejor_mcr_dinamico))
	x_data = range(0, t, freq_output)
	if u_t is not None:
		x_data = [1-u_t[ii] for ii in x_data]
	plt.plot(x_data, mcr_lista, "black")
	plt.xlabel('Time (%s)'%unidades)
	plt.ylabel('MSE')
	# plt.title('Mean Squared Error as a function of time')
	axes = plt.gca()
	axes.set_ylim([0,None])
	if mostrar:
		plt.show()
	else:
		plt.savefig('out/an_din_1.png', bbox_inches='tight')
		plt.clf()
	factor = [1.4, 1.7, 2.]
	colores = ["green", "blue", "red"]
	leyenda = [mpatches.Patch(color='black', label="C=%.2f"%mejor_c)]
	plt.plot(x_data, mcr_lista, "black")
	for ii in range(3):
		nuevo_c = mejor_c*factor[ii]
		mcr_lista_k = mydb.mcr_dinamico(stoc_cdf, cdf = f_X, l_factor = l_factor, damp=damp, c_factor = nuevo_c, t = t, intervalo = freq_output, u_t = u_t)
		print( "C=%.2f, MCR=%.5f"%(nuevo_c, sum(mcr_lista_k)/len(mcr_lista_k)))
		plt.plot(x_data, mcr_lista_k, colores[ii])
		leyenda.append(mpatches.Patch(color=colores[ii], label="C=%.2f"%nuevo_c))
	plt.xlabel('Time (%s)'%unidades)
	plt.ylabel('MSE')
	# plt.title('Error Cuadratico Medio en funcion del tiempo')
	plt.legend(handles=leyenda)
	axes = plt.gca()
	axes.set_ylim([0,None])
	if mostrar:
		plt.show()
	else:
		plt.savefig('out/ecm_%s_1.png'%deporte, bbox_inches='tight')
		plt.clf()
	leyenda = [mpatches.Patch(color='black', label="C=%.2f"%mejor_c)]
	plt.plot(x_data, mcr_lista, "black")
	for ii in range(3):
		nuevo_c = mejor_c/factor[ii]
		mcr_lista_k = mydb.mcr_dinamico(stoc_cdf, cdf = f_X, l_factor = l_factor, damp=damp, c_factor = nuevo_c, t = t, intervalo = freq_output, u_t = u_t)
		print( "C=%.2f, MCR=%.5f"%(nuevo_c, sum(mcr_lista_k)/len(mcr_lista_k)))
		plt.plot(x_data, mcr_lista_k, colores[ii])
		leyenda.append(mpatches.Patch(color=colores[ii], label="C=%.2f"%nuevo_c))
	plt.xlabel('Time (%s)'%unidades)
	plt.ylabel('MSE')
	# plt.title('Error Cuadratico Medio en funcion del tiempo')
	plt.legend(handles=leyenda)
	axes = plt.gca()
	axes.set_ylim([0,None])
	if mostrar:
		plt.show()
	else:
		plt.savefig('out/ecm_%s_2.png'%deporte, bbox_inches='tight')
		plt.clf()
	print( "Calculando matriz de confusion...")
	matriz = [[[0]*t for bb in range(2)] for aa in range(2)]
	for a,b,c,d,s,e1,e2,g in db_test.games:
		if s != 0.0 and s != 1.0:
			continue
		score = 0
		jj = 0
		for ii in range(0,t,freq_output):
			if jj<len(g):
				tiempo, res = g[jj].split(";")
				while int(tiempo)<=ii:
					s1,s2 = map(int, res.split("-"))
					score = s1-s2
					jj+=1
					if jj >= len(g):
						break
					tiempo, res = g[jj].split(";")
			aux = f_X((l_factor+(e1-e2)*damp)*(((t-ii+0.0)/t)**0.5) + mejor_c*score/((t-ii+0.1)/t)**0.5)
			prediccion = 0 + (0.5 < aux)
			matriz[prediccion][int(s+0.01)][ii] += 1
	for aa in range(2):
		for bb in range(2):
			plt.plot(x_data, [matriz[aa][bb][oo] for oo in range(0,t,freq_output)])
			if mostrar:
				plt.show()
			else:
				plt.savefig('out/mat_conf_%d_%d_%s.png'%(aa, bb, deporte), bbox_inches='tight')
				plt.clf()
	db_test.caminos_est(mejor_l, mejor_c, muestra = 10, t=t, deporte = deporte)
	db_test.test_sesgo_score(mejor_l, mejor_c, t=t, deporte = deporte, u_t=u_t)
	db_test.test_normal(mejor_l, mejor_c, damp=damp, deporte = deporte)
	return mejor_c

#Elaboracion de todo el analisis del modelo discreto
def analisis_discreto(archivo, archivo_test, t = 90, u_t = None, muestra = 12):
	global A
	print( "\n{*} Analisis discreto")
	f = open(archivo, "r")
	lines = f.read().split('\n')[:-1]
	f.close()
	a_cuadrado = 0
	for line in lines:
		temp, fecha, j1, j2, s1, s2 = line.split("\t")[:6]
		a_cuadrado += int(s1)*int(s2) # int(s1)**0.5+int(s2)**0.5 # 
	a_estim = 2*((a_cuadrado/	(0.0+len(lines)))**0.5) # a_cuadrado/len(lines) # 
	print( "A =", a_estim)
	A = a_estim
	#mejor_k, mejor_l = 0.15, 0.68
	# base_datos, mejor_k, mejor_l, mejor_d = 0,0.10588,0.38986,0.97036
	base_datos, mejor_k, mejor_l, mejor_d = 0,0.14781,0.62,0.86536
	# base_datos, mejor_k, mejor_l, mejor_d = analisis_estatico(archivo, archivo_test, sk_cdf, muestra = muestra,
	# 	inter_k = [0.,0.3], inter_l = [0.4,1.], deporte = "futbol_discreto", tol_1 = 0.001, tol_2 = 0.1)
	mydb = Db(rated_db(archivo_test, sk_cdf, k_factor = mejor_k, l_factor = mejor_l, muestra = muestra))
	list_sq_error = mydb.all_sq_error(sk_cdf, l_factor = mejor_l, damp=mejor_d)

	print('\n SPI system comparison')
	mejor_lambda, damp_spi = 0.02, 0.89
	aux_f = lambda x: spi_mse(archivo, lambda_factor = x[0], muestra = muestra, damp = x[1])
	optimal, mejor_mcr, iters, flag = conjugate_gradient_descent(aux_f, np.array([0.1, 0.9]), tolerance=0.001)
	mejor_lambda, damp_spi = optimal
	print ('Optimal SPI lambda = %.3f'%mejor_lambda, '   D = %.3f'%damp_spi)
	spi_rated_games, avg_goals_home, avg_goals_away = spi_rated_db(archivo_test, lambda_factor = mejor_lambda, muestra = muestra)
	mse_spi = []
	# print( len(spi_rated_games), len(mydb.games))
	# print (spi_rated_games[:5] + spi_rated_games[-5:], mydb.games[:5] + mydb.games[-5:])
	for game in spi_rated_games:
		result, off1, def1, off2, def2 = game[4:9]
		goals1, goals2, exp_score = expected_score_spi(off1*damp_spi, def1*damp_spi, off2*damp_spi, def2*damp_spi, avg_goals_home, avg_goals_away)
		# print (sum(mse_spi), game[:4], goals1, goals2, exp_score, off1, def1, off2, def2)
		mse_spi.append( (exp_score - result)*(exp_score - result) )
	print ('MSE of SPI: ', sum(mse_spi)/len(mse_spi))
	print ('MSE of DiscreteElo: ', sum(list_sq_error)/len(list_sq_error))
	print ("p_value SPI vs. DiscreteElo = ", different_mean_test(list_sq_error, mse_spi))

	print( "Computing SPI vs Elo for every game and time...")
	freq_output = max(1,int(t/100.))
	matriz = [[[0]*(t+1) for bb in range(3)] for aa in range(3)]
	p_matriz = [[[0.]*(t+1) for bb in range(3)] for aa in range(3)]
	mse_vs_time = [0.]*(t+1)
	mse_vs_time_spi = [0.]*(t+1)
	logloss_vs_time = [0.]*(t+1)
	logloss_vs_time_spi = [0.]*(t+1)
	logloss_list = []
	logloss_list_spi = []
	logloss_list_2 = []
	logloss_list_spi_2 = []
	for ii in range(len(mydb.games)):
		season1,date1,t1,t2,s,e1,e2,g = mydb.games[ii]
		season2,date2,p1,p2,ss,off1,def1,off2,def2 = spi_rated_games[ii][:9]
		if date1 != date2 or t1 != p1 or t2 != p2:
			print('Error in SPI comparison: games differ', mydb.games[ii], spi_rated_games[ii][:9])
		resultado = int(s+0.51) + (abs(s-0.5)<0.001)
		score = 0
		jj = 0
		mu = mejor_d*(e1-e2) + mejor_l
		# SQRT skellam
		# mu1 = ((mu/a_estim+a_estim)/2.)**2
		# mu2 = ((a_estim - mu/a_estim)/2.)**2
		# Discrete Elo
		mu1 = 0.5*(mu + (mu*mu + a_estim*a_estim)**0.5)
		mu2 = 0.5*(-mu + (mu*mu + a_estim*a_estim)**0.5)
		for ii in range(0,t+1,freq_output):
			if jj<len(g):
				tiempo, res = g[jj].split(";")
				while int(tiempo)<=ii:
					s1,s2 = map(int, res.split("-"))
					score = s1-s2
					jj+=1
					if jj >= len(g):
						break
					tiempo, res = g[jj].split(";")
			u = (t - ii + 1e-6)/(t+1e-6) if u_t is None else u_t[ii]
			prob_1 = skellam.cdf(score-1, u*mu2, u*mu1)
			prob_0 = skellam.cdf(-1-score, u*mu1, u*mu2)
			prob_empate = skellam.pmf(score, u*mu2, u*mu1)
			logloss_elo = -np.log2([prob_0, prob_1, prob_empate][resultado])
			logloss_vs_time[ii] += logloss_elo
			exp_score = prob_1 + 0.5*prob_empate
			mse_vs_time[ii] += (exp_score - s)**2

			p_matriz[0][resultado][ii] += prob_0
			p_matriz[1][resultado][ii] += prob_1
			p_matriz[2][resultado][ii] += prob_empate
			if abs(prob_empate+prob_0+prob_1-1.)>0.001:
				print( "ERROR: Probabilities don't add up to 1" )
			maxi = max(prob_0, prob_1, prob_empate)
			prediccion = [prob_0, prob_1, prob_empate].index(maxi)
			matriz[prediccion][resultado][ii] += 1

			goals1, goals2, exp_score = expected_score_spi(off1*mejor_d, def1*mejor_d, off2*mejor_d, def2*mejor_d, avg_goals_home, avg_goals_away)
			prob_1 = skellam.cdf(score-1, u*goals2, u*goals1)
			prob_0 = skellam.cdf(-1-score, u*goals1, u*goals2)
			prob_empate = skellam.pmf(score, u*goals2, u*goals1)
			if abs(prob_empate+prob_0+prob_1-1.)>0.001:
				print( "ERROR: Probabilities don't add up to 1" )
			logloss_spi = -np.log2([prob_0, prob_1, prob_empate][resultado])
			logloss_vs_time_spi[ii] += logloss_spi
			if ii == 0:
				logloss_list.append(logloss_elo)
				logloss_list_spi.append(logloss_spi)
			if ii == 72:
				logloss_list_2.append(logloss_elo)
				logloss_list_spi_2.append(logloss_spi)
			exp_score = prob_1 + 0.5*prob_empate
			mse_vs_time_spi[ii] += (exp_score - s)**2

	print ('Avg. log loss Elo =', sum(logloss_list)/len(logloss_list))
	print ('Avg. log loss SPI =', sum(logloss_list_spi)/len(logloss_list_spi))
	print ('p-value logloss Elo vs. SPI: ', different_mean_test(logloss_list_spi, logloss_list))

	print ('Middle log loss Elo =', sum(logloss_list_2)/len(logloss_list_2))
	print ('Middle log loss SPI =', sum(logloss_list_spi_2)/len(logloss_list_spi_2))
	print ('p-value mid. logloss Elo vs. SPI: ', different_mean_test(logloss_list_spi_2, logloss_list_2))

	x_data = range(0,t+1,freq_output) if u_t is None else [1-u_t[ii] for ii in range(0,t+1,freq_output)]
	for ii in range(0,t+1,freq_output):
		logloss_vs_time[ii] /= len(mydb.games)
		logloss_vs_time_spi[ii] /= len(mydb.games)
		mse_vs_time[ii] /= len(mydb.games)
		mse_vs_time_spi[ii] /= len(mydb.games)
	plt.plot(x_data, logloss_vs_time, color='k', label='Discrete Elo')
	plt.plot(x_data, logloss_vs_time_spi, color='r', label='SPI')
	plt.xlabel('1-u_t')
	plt.ylabel('Log-Loss')
	plt.legend()
	plt.savefig('out/logloss_time_comp.png', bbox_inches='tight')
	plt.clf()

	plt.plot(x_data, mse_vs_time, color='k', label='Discrete Elo')
	plt.plot(x_data, mse_vs_time_spi, color='r', label='SPI')
	plt.ylabel('MSE(t)')
	plt.xlabel('1-u_t')
	plt.legend()
	plt.savefig('out/mse_time_comp.png', bbox_inches='tight')
	plt.clf()

	print( "Calculando matriz de confusion 3x3...")
	for aa in range(3):
		for bb in range(3):
			plt.plot(range(0,t+1,freq_output), [matriz[aa][bb][oo] for oo in range(0,t+1,freq_output)])
			plt.savefig('out/mat_conf_%d_%d_%s.png'%(aa, bb, "D"), bbox_inches='tight')
			plt.clf()
			plt.plot(range(0,t+1,freq_output), [p_matriz[aa][bb][oo] for oo in range(0,t+1,freq_output)])
			plt.savefig('out/p_mat_conf_%d_%d_%s.png'%(aa, bb, "D"), bbox_inches='tight')
			plt.clf()

#Implementacion del test de verosimilitud usando la regresion logistica
def test_verosimilitud(partidos):
	muestra = []
	jugadores = []
	total = 0
	for partido in partidos:
		temp, fecha, jug1, jug2, s1, s2 = partido[:6]
		muestra.append((jug1, jug2, s1>s2))
		if jug2 not in jugadores:
			jugadores.append(jug2)
		if jug1 not in jugadores:
			jugadores.append(jug1)
		total += s1>s2
	p_media = total / (0.0 + len(muestra))
	log_verosim_0 = total*math.log(p_media) + (len(muestra)-total)*math.log(1-p_media)
	print( "log(verosim_0) =", log_verosim_0)
	X = [[(jj == a) - (jj == b) for jj in jugadores[1:]] for a,b,c in muestra]
	y = [c for a,b,c in muestra]
	clf = LogisticRegression().fit(X, y)
	print( "Precision:",clf.score(X, y))
	predicho = clf.predict_proba(X)
	print( "coeficientes:",clf.coef_)
	print( "termino indep.:",clf.intercept_)
	log_verosim_1 = 0.0
	for ii in range(len(y)):
		log_verosim_1 += math.log(predicho[ii][y[ii]+0])
	print( "log(verosim_1) =", log_verosim_1)
	print( "G = 2(log(v_1)-log(v_0)) =", 2*(log_verosim_1 - log_verosim_0))
	print( "grados de libertad:", len(jugadores)-1)
	print( "Repitiendo test para menos grados de libertad...")
	variables = filter(lambda x: abs(clf.coef_[0][x])>0.3, range(len(jugadores)-1))
	mis_jugadores = [jugadores[ii+1] for ii in variables]
	X = [[(jj == a) - (jj == b) for jj in mis_jugadores] for a,b,c in muestra]
	clf = LogisticRegression().fit(X, y)
	print( "Precision:",clf.score(X, y))
	predicho = clf.predict_proba(X)
	print( "coeficientes:",clf.coef_)
	print( "termino indep.:",clf.intercept_)
	log_verosim_1 = 0.0
	for ii in range(len(y)):
		log_verosim_1 += math.log(predicho[ii][y[ii]+0])
	print( "log(verosim_1) =", log_verosim_1)
	print( "G = 2(log(v_1)-log(v_0)) =", 2*(log_verosim_1 - log_verosim_0))
	print( "grados de libertad:", len(mis_jugadores))

#Funcion para hacer graficos con regresion lineal
def grafico_reg_lineal(x,y, archivo, modo="plot"):
	gradient,intercept,r_value,p_value,std_err=linregress(x,y)
	if modo == "plot":
		plt.plot(x,y)
	elif modo == "scatter":
		plt.scatter(x,y, s=1)
	plt.plot(x, [gradient*t+intercept for t in x], "k", linewidth = 1)
	plt.legend(handles = [mlines.Line2D([], [], color = "black", label = "r² = %.5f"%(r_value**2)),
		mlines.Line2D([], [], color = "black", label = "slope = %.2f"%gradient),
		mlines.Line2D([], [], color = "black", label = "intercept = %.2f"%intercept)])
	plt.savefig(archivo, bbox_inches='tight')
	plt.clf()
	return gradient,intercept,r_value,p_value,std_err


if __name__ == "__main__":
	# mejor_k, mejor_l = 15.95, -60.6
	# mejor_k, mejor_l, mejor_d = 52.500, 11.822, 0.874
	#Calculo de F_X basada en la distribucion skellam para distintos valores de A
	# xlis = [it / 100. for it in range(-500, 501)]
	# for aa in [.1, 1., 10.]:
	# 	A = aa
	# 	plt.plot(xlis, [sk_cdf(it) for it in xlis])
	# plt.savefig("out/f_skellam_a.png")
	# plt.clf()

	base_datos, mejor_k, mejor_l, mejor_d = analisis_estatico("traindb.txt", "testdb.txt", n_cdf, muestra = 20, deporte = "basket")
	base_datos_test = Db(rated_db("testdb.txt", cdf = n_cdf, muestra = 20, k_factor = mejor_k, l_factor = mejor_l))
	mejor_c = analisis_dinamico(base_datos, base_datos_test, n_cdf, mejor_l, damp=mejor_d, t = 2880, intervalo = 240, unidades="segundos", deporte = "basket")

	base_datos, mejor_k, mejor_l, mejor_d = analisis_estatico("traindf.txt", "testdf.txt", n_cdf, muestra = 12, deporte = "futbol")
	base_datos_test = Db(rated_db("testdf.txt", n_cdf, k_factor = mejor_k, l_factor = mejor_l))
	mejor_c = analisis_dinamico(base_datos, base_datos_test, n_cdf, mejor_l, damp=mejor_d, t = 92, intervalo = 5, deporte = "futbol")

	u_function = time_analysis(['testdf.txt'], saveplt='out/score_frequency_soccer.png')
	mejor_c = analisis_dinamico(base_datos, base_datos_test, n_cdf, mejor_l, damp=mejor_d, t = 90, intervalo = 5, deporte = "futbol_corrected", u_t = u_function)
	analisis_discreto("traindf.txt", "testdf.txt", u_t=u_function, muestra = 12)


	### Logistic test
	# f = open("traindb.txt", "r")
	# partidos = filter(lambda x: x[0] == "2008-2009", [part.split("\t") for part in f.read().split("\n")])
	# f.close()
	# print( "\n{*} Test de verosimilitud logistico")
	# test_verosimilitud(partidos)
	# sys.exit()


	# base_datos = Db(rated_db("traindb.txt", cdf = n_cdf, muestra = 20, k_factor = mejor_k, l_factor = mejor_l) + rated_db("testdb.txt", cdf = n_cdf, muestra = 20, k_factor = mejor_k, l_factor = mejor_l))
	# base_datos.result_vs_rating_diff(n_cdf, l_factor=mejor_l)

	# u_function = time_analysis(['testdb.txt'], rounding = 10, scaling = 60)#, saveplt='out/score_frequency_basket.png')
	

